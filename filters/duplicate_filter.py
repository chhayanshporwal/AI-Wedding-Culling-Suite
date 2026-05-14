# filters/duplicate_filter.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict

class DuplicateFilter:
    """
    Cluster images by L2 distance on CLIP embeddings using sklearn NearestNeighbors.
    Images within `threshold` are considered duplicates.
    """

    def __init__(self, threshold: float = 0.3):
        """
        threshold: max L2 distance to consider two images duplicates
        """
        self.threshold = threshold
        self.embs: List[np.ndarray] = []
        self.paths: List[str] = []

    def add(self, path: str, emb: np.ndarray):
        """Add one image and its embedding."""
        self.paths.append(path)
        self.embs.append(emb.astype("float32"))

    def cluster(self) -> List[List[str]]:
        """
        Find connected groups where any pair within threshold are duplicates.
        Returns: list of groups, each group is a list of file paths.
        """
        n = len(self.embs)
        if n == 0:
            return []

        # Stack embeddings
        data = np.vstack(self.embs)  # shape (n, dim)

        # Build radius neighbors graph
        nbrs = NearestNeighbors(radius=self.threshold,
                                metric="euclidean",
                                n_jobs=-1).fit(data)
        graph = nbrs.radius_neighbors_graph(data, mode="connectivity").tolil()

        # DFS to extract connected components
        visited = set()
        groups: List[List[str]] = []

        for i in range(n):
            if i in visited:
                continue
            stack = [i]
            comp = []
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.append(u)
                for v in graph.rows[u]:
                    if v not in visited:
                        stack.append(v)
            groups.append([self.paths[idx] for idx in comp])

        return groups
