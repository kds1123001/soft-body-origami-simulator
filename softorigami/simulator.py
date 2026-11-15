from dataclasses import dataclass
import torch
from .energy import elastic_energy, vertex_face_barrier, ground_barrier

@dataclass
class SimulationConfig:
    dt: float = 2e-4
    steps: int = 100
    mass: float = 1.0
    damping: float = 0.08
    stretch_stiffness: float = 1500.0
    area_stiffness: float = 400.0
    hinge_stiffness: float = 2.0
    collision_stiffness: float = 1500.0
    collision_thickness: float = 0.01
    ground_height: float = -1.0
    ground_stiffness: float = 1500.0
    max_force: float = 1e5

@dataclass
class SimulationResult:
    positions: torch.Tensor
    velocities: torch.Tensor
    trajectory: torch.Tensor
    energy: torch.Tensor

class DifferentiableSimulator:
    def __init__(self, mesh, config=None, pinned_vertices=None):
        self.mesh = mesh
        self.config = config or SimulationConfig()
        self.edges, self.rest_lengths, self.rest_areas = mesh.rest_data()
        self.hinges = mesh.topology()
        self.pins = torch.as_tensor(
            [] if pinned_vertices is None else pinned_vertices,
            dtype=torch.long, device=mesh.device)
        self.pin_targets = mesh.vertices[self.pins].clone()
        incidence = torch.zeros((mesh.vertex_count, mesh.faces.shape[0]),
                                dtype=torch.bool, device=mesh.device)
        if mesh.faces.numel():
            face_ids = torch.arange(mesh.faces.shape[0], device=mesh.device)
            for k in range(3):
                incidence[mesh.faces[:, k], face_ids] = True
        self.incidence = incidence

    def potential(self, x, rest_angles):
        c = self.config
        value = elastic_energy(x, self.edges, self.rest_lengths,
                               self.mesh.faces, self.rest_areas,
                               self.hinges, rest_angles, c)
        value = value + vertex_face_barrier(
            x, self.mesh.faces, c.collision_thickness,
            c.collision_stiffness, self.incidence)
        value = value + ground_barrier(
            x, c.ground_height, c.collision_thickness, c.ground_stiffness)
        if self.pins.numel():
            value = value + 0.5 * 1e5 * ((x[self.pins]-self.pin_targets)**2).sum()
        return value

    def simulate(self, rest_angles, initial_positions=None, initial_velocities=None,
                 steps=None, create_graph=True, record_trajectory=True):
        c = self.config
        count = c.steps if steps is None else int(steps)
        x = self.mesh.vertices if initial_positions is None else initial_positions
        v = torch.zeros_like(x) if initial_velocities is None else initial_velocities
        states, energies = [x], []
        for _ in range(count):
            x_eval = x if x.requires_grad else x.detach().requires_grad_(True)
            energy = self.potential(x_eval, rest_angles)
            (grad_x,) = torch.autograd.grad(
                energy, x_eval, create_graph=create_graph,
                retain_graph=create_graph)
            force = -grad_x
            scale = (c.max_force / force.norm().clamp_min(c.max_force)).clamp(max=1.0)
            force = force * scale
            v = (v + c.dt * force / c.mass) * max(0.0, 1.0-c.damping*c.dt)
            x = x_eval + c.dt * v
            if self.pins.numel():
                mask = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
                mask[self.pins] = 0
                x = x*mask + self.pin_targets*(1-mask)
                v = v*mask
            energies.append(energy)
            if record_trajectory:
                states.append(x)
        trajectory = torch.stack(states) if record_trajectory else x.unsqueeze(0)
        energy_trace = torch.stack(energies) if energies else x.new_empty((0,))
        return SimulationResult(x, v, trajectory, energy_trace)
