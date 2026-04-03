from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class IndexArtifacts:
    model_name: str
    embedding_dim: int
    n_candidates: int
    text_column: str


class CandidateIndexer:
    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_column: str = "combined_text",
        normalize_embeddings: bool = False,
    ) -> None:
        self.model_name = model_name
        self.text_column = text_column
        self.normalize_embeddings = normalize_embeddings

        self._model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.artifacts: Optional[IndexArtifacts] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, candidates_df: pd.DataFrame, *, batch_size: int = 64) -> "CandidateIndexer":
        if self.text_column not in candidates_df.columns:
            raise ValueError(
                f"Expected column '{self.text_column}' in candidates_df. "
                f"Available columns: {list(candidates_df.columns)}"
            )

        texts = candidates_df[self.text_column].fillna("").astype(str).tolist()

        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype("float32")

        if emb.ndim != 2:
            raise RuntimeError(f"Unexpected embedding shape: {emb.shape}")

        dim = int(emb.shape[1])
        index = faiss.IndexFlatL2(dim)
        index.add(emb)

        self.index = index
        self.embeddings = emb
        self.artifacts = IndexArtifacts(
            model_name=self.model_name,
            embedding_dim=dim,
            n_candidates=int(emb.shape[0]),
            text_column=self.text_column,
        )
        return self

    def search(self, query: str, *, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("FAISS index is not built/loaded. Call build() or load_index() first.")

        q = str(query or "")
        q_emb = self.model.encode(
            [q],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype("float32")

        distances, indices = self.index.search(q_emb, int(top_k))
        return distances[0], indices[0]

    def save_index(self, path: str | Path) -> None:
        if self.index is None or self.embeddings is None or self.artifacts is None:
            raise RuntimeError("Nothing to save. Build or load an index first.")

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(out_dir / "faiss.index"))
        np.save(out_dir / "embeddings.npy", self.embeddings)

        meta = asdict(self.artifacts)
        meta["normalize_embeddings"] = bool(self.normalize_embeddings)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load_index(cls, path: str | Path) -> "CandidateIndexer":
        in_dir = Path(path)
        meta_path = in_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json at: {meta_path}")

        meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))

        obj = cls(
            model_name=str(meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            text_column=str(meta.get("text_column", "combined_text")),
            normalize_embeddings=bool(meta.get("normalize_embeddings", False)),
        )

        index_path = in_dir / "faiss.index"
        emb_path = in_dir / "embeddings.npy"

        if not index_path.exists():
            raise FileNotFoundError(f"Missing faiss index file at: {index_path}")
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embeddings file at: {emb_path}")

        obj.index = faiss.read_index(str(index_path))
        obj.embeddings = np.load(emb_path)

        obj.artifacts = IndexArtifacts(
            model_name=obj.model_name,
            embedding_dim=int(meta.get("embedding_dim", obj.embeddings.shape[1])),
            n_candidates=int(meta.get("n_candidates", obj.embeddings.shape[0])),
            text_column=obj.text_column,
        )
        return obj
