from typing import List, Dict, Any
from datasets import Dataset
from transformers import AutoModel
import numpy as np

class EmbeddingStore:
  def __init__(self, ds: Dataset, embedding_model: str):
    self.ds = ds

    self.model = AutoModel.from_pretrained(embedding_model, trust_remote_code=True).to('cuda')
    self.model.eval()

    self.texts = ds["text"]
    self.labels = ds["label"]

    self.embeddings = self.model.encode(
      self.texts,
    )

  def get_k_nearest(self, query: str, k: int, out_of: int = 10) -> List[Dict[str, Any]]:
    query_embedding = self.model.encode(query)
    distances = self.embeddings @ query_embedding
    indices = distances.argsort()[-out_of:]
    indices = np.random.choice(indices, size=k, replace=False)

    return [
      {
        "review": self.texts[i],
        "classification": self.labels[i],
      }
      for i in indices
    ]
  
  def get_k_nearest_batched(self, queries: List[str], k: int, out_of: int = 10) -> List[List[Dict[str, Any]]]:
    query_embeddings = self.model.encode(queries)
    distances = self.embeddings @ query_embeddings.T
    indices = distances.argsort(axis=0)[-out_of:]
    indices = [
        np.random.choice(indices[:, j], size=k, replace=False)
        for j in range(len(queries))
    ]
    return [
      [
        {
          "review": self.texts[i],
          "classification": self.labels[i],
        }
        for i in indices[j]
      ]
      for j in range(len(queries))
    ]

  