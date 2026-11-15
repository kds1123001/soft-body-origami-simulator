import numpy as np
import torch
from softorigami import OrigamiMesh, DifferentiableSimulator, SimulationConfig

def test_unrolled_gradient_reaches_crease_angle():
    verts = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.]])
    mesh = OrigamiMesh.create(verts, [[0,1,2],[0,2,3]], [[0,2]])
    cfg = SimulationConfig(steps=2, dt=1e-4, ground_height=-2.0,
                           collision_stiffness=0.0, ground_stiffness=0.0)
    sim = DifferentiableSimulator(mesh, cfg, pinned_vertices=[0])
    angle = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    result = sim.simulate(angle, create_graph=True)
    result.positions.square().sum().backward()
    assert angle.grad is not None
    assert torch.isfinite(angle.grad).all()

def test_trajectory_shape():
    verts = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.]])
    mesh = OrigamiMesh.create(verts, [[0,1,2],[0,2,3]], [[0,2]])
    sim = DifferentiableSimulator(mesh, SimulationConfig(steps=2, ground_height=-2.0))
    result = sim.simulate(torch.zeros(1, dtype=torch.float64))
    assert result.trajectory.shape == (3, 4, 3)
