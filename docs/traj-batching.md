# Batched EMRI trajectories: status and the GPU port

Branch `traj-batch-gpu` (FEW), from `gpu_backend` @ `65a2e0ef`.

## What exists (CPU reference, tested)

| Piece | Where | Check |
|---|---|---|
| `KerrEccEqFlux.evaluate_rhs_batch(y (6,S), a (S,))` -> `(ydot, status)` | `trajectory/ode/flux.py` | == scalar `__call__` per column, rtol 1e-12, mixed spins |
| numba tricubic from multispline's own coefficients | `tricubic_grid`, `tricubic_eval_batch` (same file) | == `TricubicSpline.__call__` to 1e-13 |
| vectorised bracketed root-finder | `utils/utility.py` `brentq_array` | == scipy brentq per element |
| `BatchedIntegrate.run(m1, m2, a, y0 (6,S), T, dt)` | `trajectory/integrate.py` | dense solution within 1-4x the scalar path's own err 1e-11 vs 1e-12 spread; systems bit-independent |
| `FewBackendConsumer` default backend fix | `utils/baseclasses.py` | `FastKerrEccentricEquatorialFlux()` with no args works |

Status codes (per system): RHS 0 ok, 1 `e < 0`, 2 inside the separatrix, 3 off the flux grid
(NaN rates instead of exceptions); integrator -1 separatrix buffer, -2 tmax, -3 failed.

## Facts the port must respect

* **Flux splines.** Four tricubics (pdot, edot x region A, B). multispline stores per CELL the
  64 monomial coefficients of the local normalised coordinates: array `(nx, ny, 64 nz)`, entry
  `64 k + 16 mx + 4 my + mz`. Region A cells 128 x 64 x 64 (268 MB per spline as float64),
  region B 64 x 32 x 32. `TricubicSpline.coefficients` COPIES the whole array out of C++ on
  every access: read it once.
* **Region switch** `p <= p_sep + DELTAPMAX` (9.001) inside `_kerrecceq_flux_forward_map`;
  `risco = separatrix(a_in, 0, 1)` is fixed per system (cache it; the scalar code root-finds it
  on every call).
* **Parity is on the solution, not the knots.** DOPR853's error estimate is at roundoff
  (`errOld` ~1e-5 of abstol 1e-11, i.e. ~1e-16 absolute): last-bit RHS differences move the
  accepted steps. Validate a GPU trajectory against the scalar DENSE output at the GPU knot
  times, within the scalar path's own err-1e-11 vs err-1e-12 spread (the CPU tests do this).
* DOPR853 keeps `errOldTemp` / `previousRejectTemp` on the instance across trajectories: use a
  fresh stepper per batch.

## CPU fan-out (the chosen route, 2026-09-30)

`few.trajectory.pool`: independent trajectories integrated on a small pool of worker
processes (spawn context by default; any `concurrent.futures.Executor`, e.g.
`mpi4py.futures.MPIPoolExecutor`), shipped back as compact snapshots (call output + the
integrator's dense-output state) and served to an unchanged waveform generator by
`TrajectoryCache`, which stands in for `waveform_gen.inspiral_generator`.
`TrajectoryCache.precompute(fn, rows, pool)` dry-runs each evaluation up to its inspiral
call (so every input convention the caller applies is captured exactly), integrates the
new calls on the pool, and stores them; the real evaluations are then cache hits.
Workers reset DOPR853's controller state before each trajectory, so a pooled trajectory is
bit-identical to a freshly built module's. Tests: `tests/test_traj_pool.py`; throughput:
`examples/bench_traj_pool.py`.

## GPU port (DEFERRED by the 2026-09-30 ruling: CPU fan-out first; needs the CUDA toolkit)

Building FEW's compiled backend in the shared `deving` env would overwrite `few_backend_cpu`
used by the main checkout. Do this on the cluster in a dedicated env.

1. **B2 `FluxInterp3D.cu/.hh`** (CPU mirror via `#ifdef __CUDACC__`): `TricubicGridView {nx, ny,
   nz, x0, dx, y0, dy, z0, dz, const double* coeffs}` uploaded once (host `new` + `cudaMemcpy`
   of the struct, sprint rule); one thread per point, `which_grid` selects the spline. Test:
   == `tricubic_eval_batch` to 1e-13 on 10k points per spline. Consider the 8-corner Hermite
   form (f, f_x, f_y, f_z, f_xy, f_xz, f_yz, f_xyz per node, ~35 MB per region-A spline)
   rebuilding the 64 cell coefficients on the fly; verify it reproduces the coefficients.
2. **B3 `EMRIRhs.cu`**: translate `_kerr_ecc_eq_rhs_batch` line by line (it already calls only
   scalar numba kernels: `_get_separatrix_kernel_inner` + `_brentq_jit`, the Mino/coordinate
   frequency chain with the Carlson elliptic integrals, `_kerrecceq_flux_forward_map`,
   `_pdot_PN`, `_edot_PN`); iteration caps return status 4 instead of raising.
   Test: == `evaluate_rhs_batch` rtol 1e-12 on 10k random in-bounds states + status columns.
3. **B4** device-resident `BatchedIntegrate`: stage buffers on `xp`, one RHS launch per stage,
   host control on the small per-system vectors only. Benchmark `numSys` in
   {1, 8, 64, 256, 1024} (trajectories/s); the plan's expectation is >= 50x the scalar CPU
   rate at 256.

## Not batched yet (downstream)

`SparseInfoHolder` and everything in `waveform/base.py` reading `integrator_spline_t` /
`integrator_spline_phase_coeff` assume one trajectory (`assert len(inds_split_all) == 1`).
