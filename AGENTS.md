# mdpy Development Guide

## Environment

- Always use `conda activate md_analysis`
- Python 3.14 / numba 0.64 / numpy 2.4 / cupy 14.0 / openmm 8.4
- Dev install: `conda develop .` (editable install, source changes take effect immediately)

## GPU-Only Design Philosophy

mdpy is a GPU-native MD engine. Five hard rules:

### 1. Data Stays on GPU

All simulation data (positions, velocities, forces, energies, tile list) resides in GPU memory for the entire simulation. CPU touches data only at two points:

- **Input**: loading structure files (PSF/PDB/PRM) → one-time CPU→GPU upload
- **Output**: calling `dump_state()` or writing trajectory frames → on-demand GPU→CPU download

Specifically:
- `compute_forces()` must NOT download forces to CPU — the integrator reads `d_forces` directly on GPU
- `check_rebuild()` must NOT download all positions — use a GPU reduction kernel to compute max displacement, return bool without CPU readback
- Energy accumulation should happen on GPU — defer `float(d_energy[0])` readback until output time
- PBC wrapping should run on GPU, not via numpy in `_wrap_and_upload()`

### 2. Hard Dependencies, No Probing

`cupy` and `numba.cuda` are hard requirements, not optional. mdpy will not run without them.

- Do NOT write `try: import cupy` / `except ImportError` fallback patterns
- Do NOT use `_HAS_CUPY` / `_HAS_GPU` / `_use_gpu` runtime probes
- Do NOT query GPU availability at import time with `cp.zeros(1)`
- Import `cupy` and `numba.cuda` unconditionally at module top level
- If the runtime lacks CUDA, let it crash with a clear ImportError — silent degradation is worse than a loud failure

### 3. No CPU Fallback Paths

There is no CPU execution path. mdpy requires a CUDA-capable GPU.

- No `_rebuild_cpu()` method in TileList
- No `if self._use_gpu` branches in GPUContext
- No pure-Python fallbacks for numba kernels
- `environment.py` does not offer `set_platform('CPU')` — platform is always CUDA
- The `@njit` CPU kernels currently used during tile list rebuild (`_build_atom_to_block_slot`, `_build_masks_numba`) are acknowledged tech debt — they will be migrated to GPU kernels

### 4. Pure-Python GPU Kernel Strategy

mdpy is a pure-Python MD engine — this is a core selling point. All GPU kernels are written through Python toolchains; no `.cu` files.

- **Default: `numba.cuda.jit`** — bonded forces, integrators, tile list kernels, PBC wrapping, all use `@cuda.jit`
- **Exception: `cupy.RawKernel` for nonbonded force only** — nonbonded is the computational hotspot (O(N²) pair interaction) and requires CUDA-C-level optimization:
  - Expression transpiler dynamically converts Python AST to CUDA C for zero-overhead kernel generation
  - Enables kernel specialization per force expression without runtime branching
- **When considering `cupy.RawKernel` outside nonbonded**: you MUST explain the reason to the user and get explicit approval before proceeding. `numba.cuda` must always be tried first
- **New module guideline**: always start with `numba.cuda.jit`; only escalate to `cupy.RawKernel` when `numba.cuda` cannot implement the required feature or cannot meet performance requirements

### 5. No GPU→CPU Transfers in Hot Path (Absolute)

Simulation data is a GPU-side black box during the run loop. The only way to observe GPU state is through explicit output calls.

The following functions — and every function they call directly or indirectly — must NOT transfer any data from GPU to CPU:

- `System.step()` call chain: `check_rebuild()`, `rebuild()`, `compute_forces()`, `integrator.step()`
- `System.minimize()` inner loop

**Forbidden operations** inside the hot path:

- `cp.asnumpy(array)` — full array download
- `array.get()` — cupy `.get()` download
- `float(device_scalar)` / `int(device_scalar)` — scalar readback (causes GPU pipeline stall)
- `cp.max()` / `cp.sum()` / `cp.min()` followed by immediate CPU readback — GPU reduction with sync

**All GPU→CPU transfers must go through explicit output APIs:**

| API | Returns | GPU transfers |
|-----|---------|---------------|
| `System.dump_state()` | `(positions, velocities)` as numpy arrays | `download_positions()` + `download_velocities()` |
| `System.dump_energy()` | `dict[str, float]` of per-term energies | `cp.asnumpy(d_energy_accumulator)` |

There are no `potential_energy` or `energies` properties on System. Energy is not "queryable state" — it is "output you explicitly request". Callers who need state during a loop collect it at explicit checkpoints:

```python
for i in range(10000):
    system.step(integrator)
    if i % 1000 == 0:
        pos, vel = system.dump_state()
        energies = system.dump_energy()
```

**Rationale**: Every GPU→CPU transfer forces the CPU to wait for the entire GPU pipeline to drain. On 95K atoms this costs ~0.5–1 ms per stall — comparable to the actual compute work. Deferring all readback to explicit output points keeps the GPU pipeline saturated.

## Naming Convention (strict)

No abbreviations. Readability first:

- `position`, `force`, `velocity`, `number_particles`, `cutoff_radius`, `temperature`
- Exception: widely recognized physics abbreviations are fine (`pbc`, `lj`, `pme`, `rmsd`, `rdf`)

## Testing

```bash
# All tests
conda run -n md_analysis pytest mdpy/test/ -sv

# Single test file
conda run -n md_analysis pytest mdpy/test/test_ensemble.py -sv

# Quick validation (6PO6, 49 atoms only)
conda run -n md_analysis pytest mdpy/test/test_openmm_validation.py -sv -k "6PO6"

# Full validation (includes 1M9Z, 95567 atoms, slower)
conda run -n md_analysis pytest mdpy/test/test_openmm_validation.py -sv -m "slow"

# Brute-force gold standard (1M9Z, must pass after any force/tile-list change)
CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis pytest mdpy/test/test_bruteforce_validation.py -sv -m slow
```

- Test framework: pytest
- `run_test.py` sets `NUMBA_DISABLE_JIT=1` (pure Python interpretation, skips JIT compilation latency)
- Test execution order is defined in `conftest.py` via `test_order`
- Test data directory: `mdpy/test/data/`

### OpenMM Correctness Validation

Uses the OpenMM Reference platform (deterministic floating-point computation) as the ground truth for force calculations.

**Reference value generation** (re-run after modifying force computation code):

```bash
conda run -n md_analysis python mdpy/test/generate_reference.py
```

Output files: `mdpy/test/data/openmm_reference_6PO6.npz` and `openmm_reference_1M9Z.npz`

**Error tolerances**:

| Force term | Energy relative error | Force component absolute error | Reason |
|------------|----------------------|-------------------------------|--------|
| bond | < 1e-6 | < 1e-4 | Analytic formula, should match exactly |
| angle | < 1e-6 | < 1e-4 | Analytic formula |
| dihedral | < 1e-6 | < 1e-4 | Analytic formula |
| improper | < 1e-6 | < 1e-4 | Analytic formula |
| nonbonded (LJ) | < 1e-3 | < 1e-2 | Cutoff/switching implementation details may differ |
| electrostatic | < 1e-3 | < 1e-2 | Direct Coulomb comparison |

**Note**: OpenMM is configured with `NoCutoff` or `CutoffNonPeriodic` to align with mdpy (mdpy has no PME).

### Brute-Force Correctness Validation (Gold Standard)

This is the **primary correctness test** for mdpy. It must pass after any change to force computation or tile list code.

Uses pure numpy O(N²) brute-force computation as ground truth. Independent of any external MD engine.

**When to regenerate reference data** (MANDATORY):
- After modifying ANY force computation code (`bonded_force.py`, `nonbonded_force.py`, `expressions/`)
- After modifying tile list code that affects neighbor pair finding or exclusion/scaling masks
- After adding new force types or modifying existing force expressions
- After changing the expression transpiler or bonded expression system

**Reference value generation:**

```bash
# Takes ~10-20 minutes (one-time computation, stores results as .npz)
CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python mdpy/test/generate_bruteforce_reference.py
```

Output file: `mdpy/test/data/bruteforce_reference_1M9Z.npz`

**Running the validation:**

```bash
# Full brute-force validation (1M9Z, 95567 atoms)
CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis pytest mdpy/test/test_bruteforce_validation.py -sv -m slow
```

**What it validates:**

| Category | Tests | What is checked |
|----------|-------|-----------------|
| Tile list | 3 tests | Neighbor pair completeness, exclusion/scaling mask correctness, atom mapping consistency |
| Bonded forces | 2 tests | Bonded energy relative error < 1e-4, total forces correlation > 0.99 |
| Bonded energies | 2 tests | Total bonded energy error < 1e-4, total energy error < 1e-3 |
| Nonbonded forces | 4 tests | Energy error < 1e-3, force direction/magnitude correlation > 0.99 |

**Key files:**

| File | Purpose |
|------|---------|
| `mdpy/test/generate_bruteforce_reference.py` | Pre-computes reference data (run once, commit .npz) |
| `mdpy/test/test_bruteforce_validation.py` | Loads reference and compares against GPU |
| `mdpy/test/data/bruteforce_reference_1M9Z.npz` | Reference data (numpy arrays) |

**Maintenance rule**: When adding new force types (e.g., CMAP, restraint), you MUST:
1. Add the corresponding brute-force computation to `generate_bruteforce_reference.py`
2. Add validation tests to `test_bruteforce_validation.py`
3. Regenerate the reference `.npz` file
4. Verify all tests pass

## GPU Kernel Profiling

### Tools

| Tool | Version | Path |
|------|---------|------|
| Nsight Compute (`ncu`) | 2025.3.1 | `/home/ubuntu/Programs/cuda/13.0/bin/ncu` |
| Nsight Systems (`nsys`) | 2025.3.2 | `/home/ubuntu/Programs/cuda/13.0/bin/nsys` |
| `nvtx` Python package | installed | `conda run -n md_analysis pip show nvtx` |

### mdpy Profiling

mdpy kernel types and their visibility in ncu/nsys:

| Kernel | Toolchain | nsys/ncu visibility |
|--------|-----------|-------------------|
| BondedForce (bond/angle/dihed/improper) | `numba.cuda.jit` | Named in timeline |
| NonbondedForce (self/cross tile) | `cupy.RawKernel` | Named in timeline |
| Verlet / Langevin integrator | `numba.cuda.jit` | Named in timeline |
| TileList (Morton/AABB/tile-find) | `cupy` / `numba.cuda` | Named in timeline |

**nsys timeline (find bottleneck kernels):**

```bash
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --output=mdpy_timeline \
  conda run -n md_analysis python benchmark/benchmark_1m9z.py
```

**ncu deep dive on a specific kernel:**

```bash
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/ncu \
  --set full \
  --launch-skip 10 --launch-count 5 \
  -k "regex:kernel_name_pattern" \
  -o mdpy_kernel_profile \
  conda run -n md_analysis python benchmark/benchmark_1m9z.py
```

### OpenMM Profiling

OpenMM CUDA platform with **PME + 12Å cutoff** (production configuration).

**Important**: mdpy has no PME (direct Coulomb with cutoff only). Total step time is NOT comparable between the two engines. Nsight profiling focuses on kernel-level metrics (occupancy, memory throughput, warp stall reasons) which are meaningful regardless of electrostatic method.

OpenMM uses pre-compiled kernels (`.cubin`/`.ptx`). ncu can report occupancy and throughput but has no source-code association.

**nsys timeline:**

```bash
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=openmm_pme_timeline \
  conda run -n md_analysis python -c "
import openmm as mm, openmm.app as app
from openmm import unit
psf = app.CharmmPsfFile('mdpy/test/data/1M9Z.psf')
pdb = app.PDBFile('mdpy/test/data/1M9Z.pdb')
params = app.CharmmParameterSet('mdpy/test/data/par_all36_prot.prm',
                                 'mdpy/test/data/toppar_water_ions.str')
psf.setBox(10.8, 10.8, 10.8)
system = psf.createSystem(params, nonbondedMethod=app.PME,
                           nonbondedCutoff=1.2*unit.nanometer)
integrator = mm.VerletIntegrator(0.002*unit.picoseconds)
sim = app.Simulation(psf.topology, system, integrator,
                      platform=mm.Platform.getPlatformByName('CUDA'),
                      platformProperties={'Precision': 'single'})
sim.context.setPositions(pdb.getPositions())
sim.context.setVelocitiesToTemperature(300*unit.kelvin)
for _ in range(10): sim.step(1)    # warmup
for _ in range(50): sim.step(1)    # profiled region
"
```

**ncu deep dive:**

```bash
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/ncu \
  --set full \
  --launch-skip 10 --launch-count 3 \
  -o openmm_kernel_profile \
  conda run -n md_analysis python -c "
# (same OpenMM script as above)
"
```

### NVTX Markers

| Package | NVTX support | Usage |
|---------|-------------|-------|
| CuPy | Built-in | `cp.cuda.nvtx.RangePush("name")` / `cp.cuda.nvtx.RangePop()` |
| Numba 0.64 | No `numba.cuda.nvtx` submodule | Use `nvtx` pip package instead |
| `nvtx` pip package | Installed | `nvtx.range_push("name")` / `nvtx.range_pop()` |

**Adding markers to benchmark scripts:**

```python
import nvtx

nvtx.range_push("bonded_force")
bonded_force.compute(gpu_context, tile_list)
nvtx.range_pop()

nvtx.range_push("nonbonded_force")
nonbonded_force.compute(gpu_context, tile_list)
nvtx.range_pop()
```

### Notes

- **6-GPU environment**: always set `CUDA_VISIBLE_DEVICES=0` (or target GPU index) to profile the correct device
- **ncu serializes the GPU**: profiling is ~10-100x slower. Always use `-k` to filter kernels and `--launch-count` to limit iterations
- **P0 violation detection**: `float(d_energy[0])` readback appears as a gap in nsys CUDA timeline — useful for locating GPU→CPU transfer violations (§5)
- **Benchmark scripts**: use existing scripts in `benchmark/` as profiling workloads

## Architecture

```
Data flow:
  PSFParser + PDBParser + CharmmTopparParser
    -> CharmmForcefield.create_topology() + create_parameter_table()
    -> System(topology, pbc_matrix, cutoff)
       -> GPUContext (all device arrays: d_positions, d_velocities, d_forces, d_energy, ...)
       -> TileList (GPU Morton sort + GPU AABB + GPU tile find; exclusion masks: CPU numba)
       -> ForceTerms: BondedForce(4x @cuda.jit) + NonbondedForce(CuPy RawKernel)
       -> Integrator: Verlet / Langevin BAOAB (@cuda.jit)
    -> GPU-only simulation loop
       -> dump_state() only when output needed
```

- **GPUContext**: central GPU memory manager, owns all `d_*` device arrays, provides upload/download/zero operations
- **TileList**: tile-based neighbor list (32 atoms per tile), GPU-accelerated build with CPU mask construction (tech debt)
- **Force terms**: `ForceTerm` base class → subclasses implement `bind()` + `compute(gpu_context, tile_list)`
- **Unit system**: internal units only — Å / dalton / fs / e / K; `mdpy.unit` restricted to I/O layer
- **Precision**: arrays use `env.NUMPY_FLOAT` / `env.NUMPY_INT`, numba signatures use matching types

## Key Files

| File | Responsibility |
|------|---------------|
| `mdpy/environment.py` | Precision config (`env.NUMPY_FLOAT`, `env.NUMPY_INT`); platform always CUDA |
| `mdpy/system.py` | Top-level simulation driver, orchestrates GPUContext + TileList + ForceTerms |
| `mdpy/core/gpu_context.py` | GPU memory manager — all `d_*` device arrays, upload/download/zero |
| `mdpy/core/tile_list.py` | Tile-based neighbor list — GPU kernels (Morton/AABB/tile-find) + CPU mask build |
| `mdpy/core/topology.py` | Molecular topology (particles/bonds/angles/dihedrals/impropers), `join()` → compact arrays |
| `mdpy/force/bonded_force.py` | 4× `@cuda.jit` kernels (bond/angle/dihedral/improper) |
| `mdpy/force/nonbonded_force.py` | CuPy RawKernel (self/cross tile) + expression transpiler |
| `mdpy/force/expressions/lennard_jones.py` | LJ expression (returns dV/dr) |
| `mdpy/force/expressions/coulomb.py` | Coulomb expression with constant `0.13893556595455` |
| `mdpy/forcefield/charmm_forcefield.py` | PSF+PDB+PRM → Topology + ParameterTable pipeline |
| `mdpy/forcefield/parameters.py` | ParameterTable with `per_type` and `per_atom` dicts |
| `mdpy/integrator/verlet.py` | `@cuda.jit` Verlet integrator |
| `mdpy/integrator/langevin.py` | `@cuda.jit` Langevin BAOAB (LCG PRNG) |
| `mdpy/io/` | File parsers (PSF/PDB/DCD/HDF5/CHARMM toppar) |
| `mdpy/test/generate_bruteforce_reference.py` | Brute-force O(N²) reference data generator for 1M9Z |
| `mdpy/test/test_bruteforce_validation.py` | Gold standard validation: tile list + forces vs brute-force |
| `benchmark/benchmark_1m9z.py` | 1M9Z performance benchmark with per-kernel GPU timing |
| `benchmark/profile_tile_list.py` | Tile list nsys profiling workload |
| `benchmark/profile_tile_list_phases.py` | Tile list per-phase timing breakdown |
| `benchmark/profile_openmm_nl.py` | OpenMM neighbor list nsys profiling workload |

## Development Notes

- All arrays default to `env.NUMPY_FLOAT` (float32) / `env.NUMPY_INT` (int32)
- GPU kernel strategy: `numba.cuda.jit` is the default for all GPU kernels; `cupy.RawKernel` is reserved exclusively for nonbonded force (computational hotspot) — see §4 Pure-Python GPU Kernel Strategy
- `Topology.join()` must be called before creating a System (`System.__init__` calls it automatically)
- After each force term's `compute()` returns, `System.compute_forces()` accumulates energy (currently per-term GPU sync — see tech debt below)
- Coulomb constant in internal units: `0.13893556595455` (= 1/(4πε₀))
- Boltzmann constant in internal units: `8.314462618e-7`
- Tile list rebuild: triggered when max atom displacement > skin/2; includes GPU Morton sort + GPU AABB + GPU tile find + CPU exclusion mask construction
- Expression transpiler: converts Python AST to CUDA C for CuPy RawKernel

## Known CPU Bottlenecks & Tech Debt

### P0 — GPU→CPU Transfer Violations (violates §5)

Every violation below must be eliminated. No new violations may be introduced.

| Location | Violation | Fix |
|----------|-----------|-----|
| `system.py:54` | `download_forces()` copies N×3 forces GPU→CPU every step, but integrator only needs `d_forces` on GPU | Remove from per-step path; only download when user requests forces |
| `bonded_force.py:427`, `nonbonded_force.py:678` | `float(d_energy[0])` reads energy GPU→CPU per force term, causing 2 pipeline stalls per step | Accumulate energy on GPU via `d_energy_accumulator`; read back only in `dump_energy()` |
| `tile_list.py:962` | `float(cp.max(diff))` in `check_rebuild()` — GPU reduction + scalar readback every step | GPU reduction kernel with deferred compare; return bool without reading scalar to CPU |
| `tile_list.py:800-802` | `cp.asnumpy(bin_coords[...])` × 3 in `_rebuild_gpu()` — downloads N×3 int32 for CPU bin aggregation | GPU bin aggregation kernel |
| `tile_list.py:896-897` | `int(counters[...])` × 2 in `_rebuild_gpu()` — scalar readback for tile counts | Device-side counters; pass to subsequent kernels on GPU |
| `tile_list.py:958` | `int(d_reverse_offset[N])` in `_build_masks_gpu()` — scalar readback for allocation size | GPU prefix sum + GPU-side allocation |
| `system.py:108-116` | `potential_energy` / `energies` properties exist on System — energy should only be accessible via `dump_energy()` | Remove properties; implement `dump_energy()` as the sole energy output API |

### P1 — CPU Computation in Hot Path

| Location | Problem | Fix |
|----------|---------|-----|
| `system.py:34-41` | `_wrap_and_upload()` does PBC wrapping with CPU numpy + 2 CPU→GPU uploads per batch | GPU PBC wrap kernel; positions never leave GPU during simulation |

### P2 — Tile List Rebuild (periodic, but very slow)

| Location | Problem | Fix |
|----------|---------|-----|
| `tile_list.py:958-987` | `_build_exclusion_masks()` runs `numba.njit` on CPU; ~23s for 95K atoms | Migrate to GPU kernel |
| `tile_list.py:685-711` | `_rebuild_gpu()` downloads bin coords GPU→CPU, processes on CPU, uploads back | Keep bin processing on GPU |

### P3 — Dead Code to Remove

| Location | What to delete |
|----------|----------------|
| `gpu_context.py:6-18` | `_HAS_CUPY` / `_HAS_GPU` / `_use_gpu` — replace with `import cupy as cp` |
| `gpu_context.py:132-166` | All `if self._use_gpu` / `else` branches — keep only GPU path |
| `tile_list.py:6-18` | `_HAS_CUPY` / `_HAS_GPU` — replace with `import cupy as cp` |
| `tile_list.py:790-956` | `_rebuild_cpu()`, `_cut_blocks()`, `_compute_block_aabbs()`, `_find_tiles()` — CPU-only fallbacks |
| `tile_list.py:359-389` | `_morton_encode()`, `_aabb_min_image_dist_sq()` — CPU-only helpers |
| `tile_list.py:370-373` | `_to_device()` conditional — simplify to `cp.asarray()` |
| `tile_list.py:402-548` | Pure Python fallbacks for `_build_atom_to_block_slot` / `_build_masks_numba` (`else` branch) |
| `environment.py:17-18,39-47` | `set_platform()`, `'CPU'` option, `supported_platforms` — platform is always CUDA |
| `nonbonded_force.py:672` | `getDeviceProperties(0)` queried every step — cache at bind time |

## Physical Constants (internal units: Å / dalton / fs / e / K)

| Constant | Value | Source |
|----------|-------|--------|
| Coulomb constant (1/4πε₀) | `0.13893556595455` | kcal·Å/(mol·e²) |
| Boltzmann constant | `8.314462618e-7` | kcal/(mol·K) in internal units |
| Conversion: OpenMM energy → mdpy | `× 1e-4` | kJ/mol → kcal/mol |
| Conversion: OpenMM force → mdpy | `× 1e-5` | kJ/(mol·nm) → kcal/(mol·Å) |
| Conversion: OpenMM position → mdpy | `× 10.0` | nm → Å |
