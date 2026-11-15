from dataclasses import dataclass
import torch

@dataclass
class OrigamiMesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    crease_edges: torch.Tensor

    @classmethod
    def create(cls, vertices, faces, crease_edges, dtype=torch.float64, device="cpu"):
        v = torch.as_tensor(vertices, dtype=dtype, device=device)
        f = torch.as_tensor(faces, dtype=torch.long, device=device)
        c = torch.as_tensor(crease_edges, dtype=torch.long, device=device)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError("crease_edges must have shape (C, 2)")
        if f.numel() and (f.min() < 0 or f.max() >= len(v)):
            raise ValueError("face index out of range")
        return cls(v, f, c)

    @property
    def vertex_count(self):
        return self.vertices.shape[0]

    @property
    def dtype(self):
        return self.vertices.dtype

    @property
    def device(self):
        return self.vertices.device

    def topology(self):
        edge_faces = {}
        for fi, tri in enumerate(self.faces.detach().cpu().tolist()):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_faces.setdefault(tuple(sorted((a, b))), []).append(fi)
        hinges = []
        for a, b in self.crease_edges.detach().cpu().tolist():
            key = tuple(sorted((a, b)))
            adjacent = edge_faces.get(key, [])
            if len(adjacent) != 2:
                raise ValueError(f"crease edge {key} must have two adjacent triangles")
            opposite = []
            for fi in adjacent:
                opposite.append(next(int(v) for v in self.faces[fi] if int(v) not in key))
            hinges.append([key[0], key[1], opposite[0], opposite[1]])
        return torch.tensor(hinges, dtype=torch.long, device=self.device).reshape(-1, 4)

    def rest_data(self):
        edges = set()
        for tri in self.faces.detach().cpu().tolist():
            edges.update(tuple(sorted(pair)) for pair in
                         ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])))
        edge_index = torch.tensor(sorted(edges), dtype=torch.long, device=self.device).reshape(-1, 2)
        lengths = torch.linalg.vector_norm(
            self.vertices[edge_index[:, 0]] - self.vertices[edge_index[:, 1]], dim=-1)
        tri = self.vertices[self.faces]
        areas = 0.5 * torch.linalg.vector_norm(
            torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1)
        return edge_index, lengths, areas
