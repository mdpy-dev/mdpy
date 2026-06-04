# mdpy Development Guide

## Environment

- Always use `conda activate md_analysis`
- Python 3.14 / numba 0.64 / numpy 2.4 / cupy 14.0 / openmm 8.4
- Dev install: `conda develop .` (editable install, source changes take effect immediately)

## GPU-Only Design Philosophy

mdpy is a GPU-native MD engine. Four hard rules:

### 1. Data Stays on GPU

All simulation data (positions, velocities, forces, energies, tile list) resides in GPU memory for the entire simulation. CPU touches data only at two points:

- **Input**: loading structure files (PSF/PDB/PRM) → one-time CPU→GPU upload
- **Output**: calling `dump_state()` or `dump_energy()` → on-demand GPU→CPU download

Specifically:
- `compute_forces()` accumulates forces on GPU via `atomicAdd` — the integrator reads `d_forces` directly on GPU
- `check_rebuild()` must NOT download all positions — use a GPU reduction kernel that writes to a device flag
- Energy accumulation happens on GPU via `d_energy_accumulator` — no `float(d_energy[0])` readback during computation
- PBC wrapping runs on GPU via `pbc_wrap_kernel`

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
- No pure-Python fallbacks for numba/cupy kernels
- `environment.py` does not offer `set_platform('CPU')` — platform is always CUDA

### 4. No GPU→CPU Transfers in Hot Path

Simulation data is a GPU-side black box during the run loop. The only way to observe GPU state is through explicit output calls.

The following functions — and every function they call directly or indirectly — must NOT transfer **bulk data** (arrays, reductions) from GPU to CPU:

- Explicit pipeline loop: `update_neighbor_list()`, `compute_forces()`, `integrator.step(system)`, `refresh_wrapped_positions()`
- `System.minimize()` inner loop

**Forbidden operations** inside the hot path:

- `cp.asnumpy(array)` — full array download
- `array.get()` — cupy `.get()` download
- `cp.max()`, `cp.sum()`, `cp.min()` followed by CPU readback — GPU reduction with sync
- Any array readback masked as "debug" or "logging"

**Allowed scalar read**: Reading a single-element device array (e.g., `int(d_counter[0])`) to determine allocation size or kernel launch parameters. This is a necessary control-flow operation that does not stall the pipeline for bulk data transfer.

**All GPU→CPU bulk data transfers must go through explicit output APIs:**

| API | Returns | GPU transfers |
|-----|---------|---------------|
| `System.dump_state()` | `(positions, velocities)` as numpy arrays | `download_positions()` + `download_velocities()` |
| `System.dump_energy()` | `dict[str, float]` of per-term energies | `cp.asnumpy(d_energy_accumulator)` |

There are no `potential_energy` or `energies` properties on System. Energy is not "queryable state" — it is "output you explicitly request". Callers who need state during a loop collect it at explicit checkpoints:

```python
system.upload_positions()
system.upload_velocities()
system.gpu.refresh_wrapped_positions()
for i in range(10000):
    system.update_neighbor_list(sync_interval=10)
    system.compute_forces()
    integrator.step(system)
    system.gpu.refresh_wrapped_positions()
    if i % 1000 == 0:
        pos, vel = system.dump_state()
        energies = system.dump_energy()
```

**Rationale**: Every GPU→CPU transfer forces the CPU to wait for the entire GPU pipeline to drain. On 95K atoms this costs ~0.5–1 ms per stall — comparable to the actual compute work. Deferring all readback to explicit output points keeps the GPU pipeline saturated.

## Single Responsibility Design

Each class does exactly one thing. This is the architectural soul of mdpy.

### TileList — spatial partitioning only

TileList provides **mapping** — it answers "which atoms are near which atoms". It owns:

- Spatial sort indices (`d_raw_order`, `d_pdb_to_sorted`, `d_sorted_to_pdb`)
- Block/tile structure (`d_block_atoms`, `d_block_center`, `d_block_size`)
- Neighbor pair maps (`d_tiles`, `d_interacting_atoms`)
- Exclusion and scaling masks (`d_exclusion_masks`, `d_scaling_masks`)
- Classified tile arrays for force kernels (`d_main_tiles`, `d_excl_tiles`, etc.)

What TileList does NOT do:
- Sort any state arrays (positions, velocities, forces) — that's GPUContext's job
- Sort any force-term data (parameters, charges, posq) — that's each force term's job
- Compute forces or energies

### GPUContext — state array owner and sorter

GPUContext owns all per-particle state arrays (\(x\), \(y\), \(z\) for positions, velocities, forces, prev_positions, wrapped_positions, masses). When TileList rebuilds and produces a new sort order, GPUContext executes the permutation of all owned arrays via `permute_state_arrays()`.

What GPUContext does NOT do:
- Build spatial neighbor lists
- Compute forces
- Sort force-term-specific data

### Force Terms — sort their own data

Each `ForceTerm` subclass owns its own parameter arrays and working buffers. When TileList rebuilds:

- **BondedForce**: remaps its own atom index arrays via `remap_indices_gpu(d_remap)`, reads sorted positions directly from GPUContext (indices are already remapped to sorted order)
- **NonbondedForce**: permutes its own parameter arrays using GPUContext's `permute_to_sorted()`, then packs its own sorted posq buffer via `pack_sorted_posq_kernel` using TileList's `d_block_atoms`

This means:
- A new force term that needs sorted data MUST implement its own sorting logic
- GPUContext provides permutation utilities (`permute_to_sorted`, `permute_state_arrays`, `permute_from_sorted`) but does NOT know about force-term-specific buffers
- TileList tells the system the sort order; it does not apply it to anything

## GPU Kernel Strategy

mdpy is a pure-Python MD engine. All GPU kernels are written through Python toolchains; no `.cu` files.

Both `cupy.RawKernel` and `numba.cuda.jit` are acceptable:

- **`cupy.RawKernel`**: CUDA C strings compiled at runtime. Used for nonbonded force (expression transpiler generates CUDA C), tile list kernels, PBC wrapping, and permutation kernels. Prefer when you need fine control over register usage, shared memory, or warp intrinsics.
- **`numba.cuda.jit`**: Python-to-PTX compilation. Used for integrators (Verlet, Langevin BAOAB). Prefer when readability of Python kernel code matters more than micro-optimization.

No hard rule about which to use. Choose based on the kernel's needs: `cupy.RawKernel` for shared memory / warp-level control, `numba.cuda.jit` for simplicity and readability. The expression transpiler (bonded and nonbonded) targets CUDA C, so those force kernels are inherently `cupy.RawKernel`.

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
| BondedForce (bond/angle/dihed/improper) | `cupy.RawKernel` | Named in timeline |
| NonbondedForce (self/cross tile) | `cupy.RawKernel` | Named in timeline |
| Verlet / Langevin integrator | `numba.cuda.jit` | Named in timeline |
| TileList (Morton/AABB/tile-find) | `cupy.RawKernel` | Named in timeline |

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
- **Benchmark scripts**: use existing scripts in `benchmark/` as profiling workloads

## Architecture

```
Data flow:
  PSFParser + PDBParser + CharmmTopparParser
    -> CharmmForcefield.create_topology() + create_parameter_table()
    -> System(topology, pbc_matrix, cutoff)
       -> GPUContext (owns all d_positions*, d_velocities*, d_forces*, d_masses*)
       -> TileList (spatial sort + block/tile structure + exclusion masks)
       -> ForceTerms: BondedForce + NonbondedForce (each owns its own params)
       -> Integrator: Verlet / Langevin BAOAB
    -> GPU-only simulation loop
       -> dump_state() / dump_energy() only when output needed

On tile list rebuild:
  1. TileList.rebuild() sorts positions via Morton code → produces d_raw_order
  2. System._permute_all_arrays() → GPUContext permutes all state arrays
  3. TileList.build_tiles() → builds neighbor pair maps and masks
  4. Per force term: bind_sorted() → each term sorts its own data
```

- **GPUContext**: owns all `d_*` state arrays; provides permutation kernels for sorting; handles PBC wrapping
- **TileList**: spatial sort (Morton code) → blocks → AABB overlap → tile pairs → exclusion masks. Provides mappings only — never sorts state arrays or force-term data
- **Force terms**: each owns its own parameter arrays. `bind_sorted()` sorts per-term data using GPUContext's permutation utilities and TileList's block structure
- **Unit system**: internal units only — Å / dalton / fs / e / K; `mdpy.unit` restricted to I/O layer
- **Precision**: arrays use `env.NUMPY_FLOAT` (float32) / `env.NUMPY_INT` (int32)

## Key Files

| File | Responsibility |
|------|---------------|
| `mdpy/environment.py` | Precision config (`env.NUMPY_FLOAT`, `env.NUMPY_INT`); platform always CUDA |
| `mdpy/system.py` | Simulation driver with public atomic operations: `upload_positions()`, `upload_velocities()`, `update_neighbor_list()`, `compute_forces()`, `dump_state()`, `dump_energy()` |
| `mdpy/core/gpu_context.py` | GPU memory manager — owns all `d_*` state arrays, permutation kernels, PBC wrap |
| `mdpy/core/tile_list.py` | Tile-based neighbor list — GPU kernels (Morton/AABB/tile-find/masks); provides mapping only |
| `mdpy/core/topology.py` | Molecular topology (particles/bonds/angles/dihedrals/impropers), `join()` → compact arrays |
| `mdpy/force/force_term.py` | `ForceTerm` base class — `compute(gpu_context, tile_list)` |
| `mdpy/force/bonded_force.py` | Single CuPy RawKernel (bond/angle/dihedral/improper); owns indices, remaps via `d_remap` |
| `mdpy/force/nonbonded_force.py` | CuPy RawKernel (self/cross tile) + expression transpiler; owns sorted posq, params |
| `mdpy/force/expressions/lennard_jones.py` | LJ expression (returns dV/dr) |
| `mdpy/force/expressions/coulomb.py` | Coulomb expression with constant `0.13893556595455` |
| `mdpy/forcefield/charmm_forcefield.py` | PSF+PDB+PRM → Topology + ParameterTable pipeline |
| `mdpy/forcefield/parameters.py` | ParameterTable with `per_type` and `per_atom` dicts |
| `mdpy/integrator/verlet.py` | `@cuda.jit` Verlet integrator (`step(system)`) |
| `mdpy/integrator/langevin.py` | `@cuda.jit` Langevin BAOAB (`step(system)`, LCG PRNG) |
| `mdpy/io/` | File parsers (PSF/PDB/DCD/HDF5/CHARMM toppar) |
| `mdpy/test/generate_bruteforce_reference.py` | Brute-force O(N²) reference data generator for 1M9Z |
| `mdpy/test/test_bruteforce_validation.py` | Gold standard validation: tile list + forces vs brute-force |
| `benchmark/benchmark_1m9z.py` | 1M9Z performance benchmark with per-kernel GPU timing |
| `benchmark/profile_tile_list.py` | Tile list nsys profiling workload |
| `benchmark/profile_tile_list_phases.py` | Tile list per-phase timing breakdown |
| `benchmark/profile_openmm_nl.py` | OpenMM neighbor list nsys profiling workload |

## Development Notes

- All arrays default to `env.NUMPY_FLOAT` (float32) / `env.NUMPY_INT` (int32)
- GPU kernel strategy: both `cupy.RawKernel` and `numba.cuda.jit` are valid; choose based on kernel needs
- `Topology.join()` must be called before creating a System (`System.__init__` calls it automatically)
- Coulomb constant in internal units: `0.13893556595455` (= 1/(4πε₀))
- Boltzmann constant in internal units: `8.314462618e-7`
- Tile list rebuild: triggered when max atom displacement > skin/2; runs fully on GPU (Morton sort + block form + AABB + tile find + mask construction + tile classification)
- Expression transpiler: converts Python AST to CUDA C for CuPy RawKernel (used by both bonded and nonbonded forces)

## Known CPU Bottlenecks & Tech Debt

### P1 — Scalar Reads in Hot Path (acceptable, but could be cleaner)

Scalar reads for kernel launch sizing and control flow. These are acceptable per §4 but are noted as targets for future device-side control flow:

| Location | Scalar read | Purpose |
|----------|------------|---------|
| `tile_list.py:927` | `int(self._d_counters[0])` | Tile count for array sizing |
| `tile_list.py:959` | `int(d_rev_offset[-1])` | Reverse map allocation size |
| `tile_list.py:1062-1063` | `int(d_classify_excl_counter[0])`, `int(d_classify_main_counter[0])` | Exclusion/main tile counts |

### P2 — Dead Code to Remove

| Location | What to delete |
|----------|----------------|
| `gpu_context.py` | Any remaining `_HAS_CUPY` / `_HAS_GPU` / `_use_gpu` if present — replace with unconditional `import cupy as cp` |
| `tile_list.py` | Any remaining `_HAS_CUPY` / `_HAS_GPU` if present — replace with unconditional `import cupy as cp` |
| `tile_list.py` | `_rebuild_cpu()`, `_cut_blocks()`, `_compute_block_aabbs()`, `_find_tiles()`, `_morton_encode()`, `_aabb_min_image_dist_sq()`, `_to_device()` — CPU-only fallbacks if still present |
| `tile_list.py` | Pure Python fallbacks for `_build_atom_to_block_slot` / `_build_masks_numba` (`else` branch) if still present |
| `environment.py` | `set_platform()`, `'CPU'` option, `supported_platforms` — platform is always CUDA |
| `nonbonded_force.py:672` | `getDeviceProperties(0)` queried every compute call — cache at bind time |

### P3 — Performance Optimizations

| Location | Problem | Fix |
|----------|---------|-----|
| `tile_list.py` tile find | `d_tile_buf` / `d_interacting_buf` are overallocated (max_tiles heuristic) | Dynamic allocation via two-pass (count then fill) |
| `nonbonded_force.py` | `_refresh_sorted_data` called every compute step, repacks sorted posq even if positions haven't changed | Skip refresh when tile list hasn't rebuilt |

## Physical Constants (internal units: Å / dalton / fs / e / K)

| Constant | Value | Source |
|----------|-------|--------|
| Coulomb constant (1/4πε₀) | `0.13893556595455` | kcal·Å/(mol·e²) |
| Boltzmann constant | `8.314462618e-7` | kcal/(mol·K) in internal units |
| Conversion: OpenMM energy → mdpy | `× 1e-4` | kJ/mol → kcal/mol |
| Conversion: OpenMM force → mdpy | `× 1e-5` | kJ/(mol·nm) → kcal/(mol·Å) |
| Conversion: OpenMM position → mdpy | `× 10.0` | nm → Å |
