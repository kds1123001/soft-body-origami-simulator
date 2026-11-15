import torch

def optimize_fold_angles(simulator, target_positions, initial_angles=None,
                         iterations=100, learning_rate=0.02,
                         simulation_steps=None, angle_regularization=1e-4):
    target = torch.as_tensor(target_positions, dtype=simulator.mesh.dtype,
                             device=simulator.mesh.device)
    if target.shape != simulator.mesh.vertices.shape:
        raise ValueError("target_positions must have shape (V, 3)")
    if initial_angles is None:
        angles = torch.zeros(simulator.hinges.shape[0], dtype=target.dtype,
                             device=target.device, requires_grad=True)
    else:
        angles = torch.as_tensor(initial_angles, dtype=target.dtype,
                                 device=target.device).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([angles], lr=learning_rate)
    history = []
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        result = simulator.simulate(angles, steps=simulation_steps,
                                    create_graph=True, record_trajectory=False)
        loss = 0.5*((result.positions-target)**2).sum()
        loss = loss + 0.5*angle_regularization*(angles**2).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([angles], 10.0)
        optimizer.step()
        with torch.no_grad():
            angles.clamp_(-torch.pi, torch.pi)
        history.append(float(loss.detach()))
    return angles.detach(), history
