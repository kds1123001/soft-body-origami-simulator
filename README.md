# Soft Origami Engine

A differentiable triangular-sheet mechanics prototype. The Python simulation unrolls the complete time integrator in PyTorch, allowing reverse-mode gradients to propagate from a terminal shape objective through every time step to crease rest angles and initial states.

## Included

- Triangular mesh stretch springs and triangle-area regularization.
- Signed dihedral hinge energies with wrapped angular residuals.
- Differentiable semi-implicit damped dynamics.
- Reverse-mode backpropagation through the full unrolled simulation.
- Smooth vertex-face self-contact barrier with adjacent-face exclusion.
- Smooth ground-contact barrier.
- Hard vertex pins.
- Adam-based target-shape optimization of crease rest angles.
- C99 geometry primitives for signed dihedral angle and edge strain.

## Install and run

```bash
python -m pip install -e .
python examples/optimize_fold.py
python -m pip install pytest
pytest -q
```

Optional native library:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Mesh conventions

`vertices` is `(V,3)`, `faces` is `(F,3)`, and `crease_edges` is `(C,2)`. Each crease edge must be shared by exactly two triangles. The mesh topology resolves the opposite vertex on each side of a crease.

## End-to-end differentiation

`DifferentiableSimulator.simulate()` keeps the computation graph connected across every integration step. At each step it computes the position gradient of potential energy with `create_graph=True`, then advances velocity and position. Calling `loss.backward()` on the final state therefore differentiates through the force calculation and the complete trajectory, including the required second derivatives of potential energy.

Memory use scales with the unrolled horizon. For large meshes or long sequences, use shorter horizons, checkpointing, or a custom discrete adjoint.

## Contact and limitations

The current self-contact implementation uses a smooth vertex-face barrier and excludes vertices incident to a face. Ground contact uses a smooth barrier. This is differentiable contact handling, not a production continuous-collision solver: edge-edge contact, swept collision detection, friction, BVH broad phase, and robust tunneling prevention are not included. Keep time steps conservative and validate gradients with finite differences.

This is a research prototype rather than a validated engineering solver. The C library provides native geometry primitives; end-to-end trajectory differentiation currently runs in PyTorch. The C trajectory solver and a custom C adjoint are not yet implemented.

## credits

-I created the project me yes me lads.
