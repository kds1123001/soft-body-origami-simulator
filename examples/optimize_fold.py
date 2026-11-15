import numpy as np
import torch
from softorigami import OrigamiMesh, DifferentiableSimulator, SimulationConfig, optimize_fold_angles

torch.set_default_dtype(torch.float64)
vertices = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.]])
faces = np.array([[0,1,2], [0,2,3]])
mesh = OrigamiMesh.create(vertices, faces, [[0,2]])
cfg = SimulationConfig(steps=60, dt=2e-4, ground_height=-0.5)
sim = DifferentiableSimulator(mesh, cfg, pinned_vertices=[0,1])
target = torch.tensor(vertices)
target[2,2], target[3,2] = 0.3, -0.3
angles, losses = optimize_fold_angles(sim, target, iterations=30, learning_rate=0.02)
print("optimized crease angles:", angles.detach().cpu().numpy())
print("loss:", losses[0], "->", losses[-1])
