import torch
import torch.nn.functional as F

def norm(x, dim=-1, keepdim=False):
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + 1e-12)

def signed_dihedral(x, hinges):
    if hinges.numel() == 0:
        return x.new_empty((0,))
    i, j, a, b = hinges.unbind(1)
    p0, p1, pa, pb = x[i], x[j], x[a], x[b]
    edge = p1 - p0
    n0 = torch.cross(edge, pa - p0, dim=-1)
    n1 = torch.cross(pb - p0, edge, dim=-1)
    n0 = n0 / norm(n0, keepdim=True)
    n1 = n1 / norm(n1, keepdim=True)
    tangent = edge / norm(edge, keepdim=True)
    sine = torch.sum(torch.cross(n0, n1, dim=-1) * tangent, dim=-1)
    cosine = torch.sum(n0 * n1, dim=-1)
    return torch.atan2(sine, cosine)

def wrap_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

def elastic_energy(x, edges, lengths, faces, areas, hinges, rest_angles, cfg):
    e = x.new_zeros(())
    if edges.numel():
        current = norm(x[edges[:, 0]] - x[edges[:, 1]])
        e = e + 0.5 * cfg.stretch_stiffness * ((current - lengths) ** 2).sum()
    if faces.numel() and cfg.area_stiffness:
        tri = x[faces]
        current = 0.5 * norm(torch.cross(tri[:, 1]-tri[:, 0],
                                          tri[:, 2]-tri[:, 0], dim=-1))
        e = e + 0.5 * cfg.area_stiffness * ((current - areas) ** 2).sum()
    if hinges.numel():
        delta = wrap_angle(signed_dihedral(x, hinges) - rest_angles)
        e = e + 0.5 * cfg.hinge_stiffness * (delta ** 2).sum()
    return e

def vertex_face_barrier(x, faces, thickness, stiffness, incidence):
    if not faces.numel() or stiffness <= 0:
        return x.new_zeros(())
    tri = x[faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b-a, c-a
    normal = torch.cross(ab, ac, dim=-1)
    normal = normal / norm(normal, keepdim=True)
    p = x[:, None, :]
    ap = p - a[None, :, :]
    d00 = (ab*ab).sum(-1).clamp_min(1e-12)
    d01 = (ab*ac).sum(-1)
    d11 = (ac*ac).sum(-1).clamp_min(1e-12)
    d20 = (ap*ab[None, :, :]).sum(-1)
    d21 = (ap*ac[None, :, :]).sum(-1)
    den = (d00*d11-d01*d01).clamp_min(1e-12)
    v = (d11[None, :]*d20-d01[None, :]*d21)/den[None, :]
    w = (d00[None, :]*d21-d01[None, :]*d20)/den[None, :]
    bary = torch.stack((1-v-w, v, w), dim=-1).clamp_min(0)
    bary = bary / bary.sum(-1, keepdim=True).clamp_min(1e-12)
    closest = (bary[..., 0:1]*a[None, :, :] +
               bary[..., 1:2]*b[None, :, :] +
               bary[..., 2:3]*c[None, :, :])
    delta = p - closest
    distance = norm(delta)
    signed = (delta * normal[None, :, :]).sum(-1).abs()
    distance = torch.maximum(distance, signed)
    penetration = F.softplus((thickness-distance)*(10.0/max(thickness, 1e-8))) * (thickness/10.0)
    penetration = penetration * (~incidence).to(x.dtype)
    return 0.5 * stiffness * (penetration ** 2).sum()

def ground_barrier(x, height, thickness, stiffness):
    if stiffness <= 0:
        return x.new_zeros(())
    penetration = F.softplus((height+thickness-x[:, 2])*(10.0/max(thickness, 1e-8)))
    penetration = penetration * (thickness/10.0)
    return 0.5 * stiffness * (penetration ** 2).sum()
