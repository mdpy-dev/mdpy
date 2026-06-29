# mdpy Development Guide

## Agent Workflow Principles

Behavioral guidelines to reduce common LLM coding mistakes. These are general-purpose; when they conflict with the project-specific hard rules below (e.g. GPU-Only Design Philosophy), **the hard rules win**. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

> Note: This principle governs code volume and scope creep — NOT the GPU-only architecture. "Add a CPU fallback because it's simpler" is forbidden by the hard rules below.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

> Exception: The "Known CPU Bottlenecks & Tech Debt" section explicitly lists items that ARE approved for removal when touched. Follow its instructions when working in those files.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

For mdpy specifically, "verified" usually means `conda run -n md_analysis pytest mdpy/test/ -sv` passes, plus `benchmark/benchmark_1m9z.py` runs clean after force/block-list changes.

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Environment

- Always use `conda activate md_analysis`
- Python 3.14 / numba 0.64 / numpy 2.4 / cupy 14.0 / openmm 8.4
- Dev install: `conda develop .` (editable install, source changes take effect immediately)

## GPU-Only Design Philosophy

mdpy is a GPU-native MD engine. Four hard rules:

### 1. Data Stays on GPU

All simulation data (positions, velocities, forces, energies, block list) resides in GPU memory for the entire simulation. CPU touches data only at two points:

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

- No `_rebuild_cpu()` method in BlockList
- No `if self._use_gpu` branches in GPUContext
- No pure-Python fallbacks for numba/cupy kernels
- `environment.py` does not offer `set_platform('CPU')` — platform is always CUDA

### 4. No GPU→CPU Transfers in Hot Path

Simulation data is a GPU-side black box during the run loop. The only way to observe GPU state is through explicit output calls.

The following functions — and every function they call directly or indirectly — must NOT transfer **bulk data** (arrays, reductions) from GPU to CPU:

- Explicit pipeline loop: `update_neighbor_list()`, `compute_forces()`, `integrator.step(system)`, `apply_constraints(dt)`
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
| `System.upload_positions(positions)` | — | CPU→GPU upload of `(N,3)` array |
| `System.upload_velocities(velocities)` | — | CPU→GPU upload of `(N,3)` array |
| `System.dump_state()` | `(positions, velocities)` as numpy arrays | `download_positions()` + `download_velocities()` |
| `System.dump_forces()` | `forces` as `(N,3)` numpy array | `download_forces()` |
| `System.dump_energy()` | `dict[str, float]` of per-term energies | `cp.asnumpy(d_energy_accumulator)` |

There are no `potential_energy` or `energies` properties on System. Energy is not "queryable state" — it is "output you explicitly request". Callers who need state during a loop collect it at explicit checkpoints:

```python
system.upload_positions(positions)       # (N,3) numpy array
system.upload_velocities(velocities)     # (N,3) numpy array
for i in range(10000):
    system.update_neighbor_list(sync_interval=10)
    system.compute_forces()
    integrator.step(system)
    system.apply_constraints(dt_fs)
    if i % 1000 == 0:
        pos, vel = system.dump_state()
        energies = system.dump_energy()
```

**Rationale**: Every GPU→CPU transfer forces the CPU to wait for the entire GPU pipeline to drain. On 95K atoms this costs ~0.5–1 ms per stall — comparable to the actual compute work. Deferring all readback to explicit output points keeps the GPU pipeline saturated.

## Single Responsibility Design

Each class does exactly one thing. This is the architectural soul of mdpy.

### BlockList — spatial partitioning only

BlockList provides **mapping** — it answers "which atoms are near which atoms". It owns:

- Spatial sort indices (`d_raw_order`, `d_pdb_to_sorted`, `d_sorted_to_pdb`)
- Block structure (`d_block_atoms`, `d_block_center`, `d_block_size`)
- Neighbor pair maps (`d_block_pairs`, exclusion/main classifications)
- Exclusion and scaling masks (`d_excl_offset`/`d_excl_neighbors`/`d_excl_scale`)
- Classified block-pair arrays for force kernels (`d_classify_excl_counter`, `d_classify_main_counter`)

What BlockList does NOT do:
- Sort any state arrays (positions, velocities, forces) — that's GPUContext's job
- Sort any force-term data (parameters, charges, posq) — that's each force term's job
- Compute forces or energies

### GPUContext — state array owner and sorter

GPUContext owns all per-particle state arrays (\(x\), \(y\), \(z\) for positions, velocities, forces, prev_positions, masses, charges). When BlockList rebuilds and produces a new sort order, GPUContext executes the permutation of all owned arrays via `permute_state_arrays()`.

What GPUContext does NOT do:
- Build spatial neighbor lists
- Compute forces
- Sort force-term-specific data

### Force Terms — sort their own data

Each `ForceTerm` owns its own parameter arrays and working buffers. When BlockList rebuilds:

- **BondedForce**: remaps its own atom index arrays via `remap_indices_gpu(d_remap)`, reads sorted positions directly from GPUContext (indices are already remapped to sorted order)
- **NonbondedForce**: permutes its own per-particle arrays using GPUContext's `permute_to_sorted()`, then packs its own sorted posq buffer via `pack_sorted_posq_kernel` using BlockList's `d_block_atoms`

This means:
- A new force term that needs sorted data MUST implement its own sorting logic
- GPUContext provides permutation utilities (`permute_to_sorted`, `permute_state_arrays`, `permute_from_sorted`) but does NOT know about force-term-specific buffers
- BlockList tells the system the sort order; it does not apply it to anything

## Force System: Expressions, Not Subclasses

**The hard rule: do NOT write a new `ForceTerm` subclass to add a force.** mdpy has exactly two force engines — `BondedForce` and `NonbondedForce` — plus the `PMEReciprocalForce` exception (see below). A new interaction is a Python *expression function* fed to one of those engines; the expression transpiler + forward-mode AD generates the CUDA kernel.

### How to add a force

| You want | What to write | Engine |
|----------|---------------|--------|
| A bonded term (bond/angle/dihed/improper or any 2–4 atom group) | `@bonded_expression(body=N)` function | `BondedForce(expr)` |
| A pairwise nonbonded term | `@nonbonded_expression` function | `NonbondedForce(expr, cutoff)` |
| Combine two nonbonded terms into one kernel | `expr1 + expr2`, or `force1 + force2` | fuses into one block-pair kernel |
| Reciprocal-space PME (FFT/grid, not pairwise) | — | `PMEReciprocalForce` (the only real subclass) |

### Expression signature conventions

The transpiler classifies function parameters by their defaults (`force/primitives.py`, `force/_utils.py`):

- **Positions**: the first `body` args (`body=2` → `pos1, pos2`; `body=3` → `p1, p2, p3`). For nonbonded, `body` is always 2.
- **Per-term scalar param**: `name=param` or `name=<number>` (e.g. `k`, `r0`, `theta0`). One value per term; passed via `BondedForce.add(indices, **params)` or `NonbondedForce.set_pair_parameter`.
- **Compile-time scalar**: `name=scalar` (e.g. PME `alpha`). One value per kernel; passed via `set_scalar`.
- **Per-particle property**: a bare arg name with no default. Trailing digits are stripped to find the base property: `charge1`/`charge2` → base `charge` on particle 1 / particle 2. Nonbonded splits per-particle into i-props (no trailing digit / `...1`) and j-props (`...2`).

### Geometry helpers (`primitives.py`)

| Helper | Bonded | Nonbonded | Returns |
|--------|--------|-----------|---------|
| `distance(a, b)` | yes | yes (must be exactly `distance(pos1, pos2)`, compiles to `r`) | scalar r |
| `angle(a, b, c)` | yes | no | scalar θ |
| `dihedral(a, b, c, d)` | yes | no | scalar φ |

The body is plain Python arithmetic over these helpers and params. `+ - * / ** %`, unary `-`, and `exp/log/sqrt/sin/cos/erf/erfc/abs/pow` are transpiled to CUDA intrinsics.

### The transpiler pipeline

1. Python AST is walked (`bonded_transpiler.py` / `nonbonded_transpiler.py`), emitting a tape of `TapeEntry(var, op, operands)` plus CUDA forward-evaluation lines.
2. `ForwardADEngine` (`ad_engine.py`) propagates derivatives forward through the tape. Power chains (e.g. `x → x² → x³ → x⁶ → x¹²`) use the power rule `d(xⁿ)/dr = n·xⁿ⁻¹·dx` with lazy emission — unreferenced intermediates are dropped to cut register pressure.
3. For bonded, each geometry helper has a hand-written forward+force CUDA template (`HELPER_REGISTRY`); the AD output becomes the scalar gradient `dE/d(geometry)`, and the template applies the chain rule to spread forces across atoms.
4. For nonbonded, the result is `energy_cuda` (forward) + `dEdr_cuda` (gradient w.r.t. `r`), assembled into self/cross block-pair and exclusion kernels.

Module-level named constants (e.g. `COULOMB_CONST`) are resolved from the decorated function's `__globals__` and inlined as CUDA float literals.

### Composition with `+`

- **Expressions fuse**: `_NonbondedExpression.__add__` merges two nonbonded expressions into one — locals of the second are suffixed `_2` to avoid collisions, energies/`dEdr` are summed. `force1 + force2` produces a `ForceGroup` that, for nonbonded, compiles a single fused `NonbondedForce` and launches **one** block-pair kernel.
- **`ForceGroup` requires homogeneous types** (`force_group.py`); you cannot mix `BondedForce` and `NonbondedForce` in one group.

### Decorate-then-override escape hatch

When the AD-generated gradient has too much register pressure or needs a faster approximation, overwrite the compiled output after decoration. See `expressions/lennard_jones.py` (closed-form `-24·ε·(2·sr¹²−sr⁶)/r`) and `expressions/screened_coulomb.py` (rational-minimax `erfc`). Pattern:

```python
@nonbonded_expression
def my_expr(pos1, pos2, ...):
    ...           # auto-compiled; output discarded

my_expr.energy_cuda = '...hand-written CUDA...'
my_expr.dEdr_cuda = '_my_force_var'
my_expr.grad_cuda = None      # no separate gradient block
my_expr._local_vars = {...}   # all float vars you declared, for renaming during fusion
```

Hand-written CUDA injects the file-local Coulomb constant via the `__MDPY_COULOMB__` placeholder (replaced at module import). Always declare `_local_vars` so the `+` fusion renamer can avoid collisions.

### The one real subclass

`PMEReciprocalForce` (`force/pme_reciprocal_force.py`) is its own `ForceTerm` subclass because reciprocal-space Ewald needs FFT/grid spread+bilerp, not a pairwise expression. It is the only force that legitimately bypasses the expression system. If a new force needs non-pairwise structure (restraints, CMAP grids), follow this precedent and justify it; otherwise use an expression.

## User-Facing Simulation Pattern

The canonical wiring (see `benchmark/benchmark_1m9z.py`):

```python
from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.integrator.langevin import LangevinBAOABIntegrator
from mdpy.system import System
from mdpy.constraint.constraint_scheme import create_constraints
from mdpy.utils import generate_velocity_from_temperature

topology = psf.topology
parameter_table = create_parameter_table(topology, toppar)
pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE

forces = create_charmm_forces(topology, parameter_table, pbc_matrix, cutoff=12.0)
# returns {'bonded': ForceGroup, 'nonbonded': NonbondedForce, 'pme': PMEReciprocalForce}

system = System(topology)          # NOT System(topology, pbc, cutoff) — pbc is uploaded separately
system.upload_pbc(pbc_matrix)
system.add_force_term(forces["bonded"])
system.add_force_term(forces["nonbonded"])
system.add_force_term(forces["pme"])

constraints = create_constraints(topology, parameter_table, scheme="h-bonds")
for c in constraints:
    system.add_constraint(c)

system.upload_positions(positions)     # (N,3) numpy array, Å
system.upload_velocities(velocities)   # (N,3) numpy array

integrator = LangevinBAOABIntegrator(dt_fs, temperature, friction)

for i in range(n_steps):
    system.update_neighbor_list(sync_interval=10)
    system.compute_forces()
    integrator.step(system)
    system.apply_constraints(dt_fs)
    if i % checkpoint == 0:
        pos, vel = system.dump_state()
        energies = system.dump_energy()
```

Key contracts:
- `System(topology)` — no pbc/cutoff in constructor. PBC via `upload_pbc()`; cutoff comes from the first force term's `_cutoff`.
- Positions **and** velocities must be uploaded before any `compute_forces()` / `update_neighbor_list()` / `minimize()` (guarded by `_ensure_uploaded()`).
- `dump_state()` / `dump_forces()` / `dump_energy()` are the **only** GPU→CPU readback paths. Call them at explicit checkpoints, never inside the step loop (see GPU-Only §4).
- The per-step order is fixed: neighbor list → forces → integrate → constraints. `apply_constraints` runs after the integrator moves positions.

## GPU Kernel Strategy

mdpy is a pure-Python MD engine. All GPU kernels are written through Python toolchains; no `.cu` files.

Both `cupy.RawKernel` and `numba.cuda.jit` are acceptable:

- **`cupy.RawKernel`**: CUDA C strings compiled at runtime. Used for nonbonded force (expression transpiler generates CUDA C), block list kernels, PBC wrapping, and permutation kernels. Prefer when you need fine control over register usage, shared memory, or warp intrinsics.
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
conda run -n md_analysis pytest mdpy/test/test_system.py -sv

# Single test by name
conda run -n md_analysis pytest mdpy/test/test_bonded_force.py -sv -k "test_name"

# OpenMM validation (6PO6, 49 atoms — quick)
conda run -n md_analysis pytest mdpy/test/test_openmm_validation.py -sv -k "6PO6"

# Performance benchmark (1M9Z, 95567 atoms)
CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/benchmark_1m9z.py

# Ion-box OpenMM validation (small pairwise system)
CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/validate_ion_openmm.py
```

- Test framework: pytest
- Test execution order is defined in `mdpy/test/conftest.py` via the `test_order` list (collect-phase reordering, not fixtures)
- Test data directory: `mdpy/test/data/`

### OpenMM Correctness Validation

`mdpy/test/test_openmm_validation.py` uses the OpenMM Reference platform (deterministic floating-point) as ground truth for force calculations.

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

**Note**: OpenMM is configured with `NoCutoff` or `CutoffNonPeriodic` for the direct-Coulomb comparison; the PME path is validated separately via `test_pme.py` / `test_pme_index_consistency.py`.

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
| NonbondedForce (self/cross block pair) | `cupy.RawKernel` | Named in timeline |
| Verlet / Langevin integrator | `numba.cuda.jit` | Named in timeline |
| BlockList (Morton/AABB/block-pair-find) | `cupy.RawKernel` | Named in timeline |

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

mdpy also uses PME (`PMEReciprocalForce` for reciprocal space + direct-space Coulomb via a screened-coulomb expression), so total step time IS broadly comparable between the two engines. Nsight profiling focuses on kernel-level metrics (occupancy, memory throughput, warp stall reasons), which are meaningful regardless.

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
bonded_force.compute(gpu_context, block_list)
nvtx.range_pop()

nvtx.range_push("nonbonded_force")
nonbonded_force.compute(gpu_context, block_list)
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
    -> create_parameter_table()
    -> System(topology)          # pbc uploaded separately via upload_pbc()
       -> GPUContext (owns all d_positions*, d_velocities*, d_forces*, d_masses*, d_charges*)
       -> BlockList (spatial sort + block structure + exclusion masks)
       -> ForceTerms: BondedForce + NonbondedForce + PMEReciprocalForce (each owns its own params)
       -> Integrator: Verlet / Langevin BAOAB
    -> GPU-only simulation loop
       -> dump_state() / dump_energy() only when output needed

On block list rebuild:
  1. BlockList.rebuild() sorts positions via Morton code → produces d_raw_order
  2. System._permute_all_arrays() → GPUContext permutes all state arrays
  3. BlockList.build_block_pairs() → builds neighbor pair maps and masks
  4. Per force term: bind_sorted() → each term sorts its own data
```

- **GPUContext**: owns all `d_*` state arrays; provides permutation kernels for sorting; handles PBC wrapping
- **BlockList**: spatial sort (Morton code) → blocks → AABB overlap → block pairs → exclusion masks. Provides mappings only — never sorts state arrays or force-term data
- **Force terms**: each owns its own parameter arrays. `bind_sorted()` sorts per-term data using GPUContext's permutation utilities and BlockList's block structure
- **Unit system**: internal units only — Å / dalton / fs / e / K; `mdpy.unit` restricted to I/O layer
- **Precision**: arrays use `env.NUMPY_FLOAT` (float32) / `env.NUMPY_INT` (int32)

## Key Files

| File | Responsibility |
|------|---------------|
| `mdpy/environment.py` | Precision config (`env.NUMPY_FLOAT`, `env.NUMPY_INT`); platform always CUDA |
| `mdpy/system.py` | Simulation driver with public atomic operations: `upload_positions(array)`, `upload_velocities(array)`, `update_neighbor_list()`, `compute_forces()`, `dump_state()`, `dump_forces()`, `dump_energy()` |
| `mdpy/core/gpu_context.py` | GPU memory manager — owns all `d_*` state arrays, permutation kernels, PBC wrap |
| `mdpy/core/block_list.py` | Block-based neighbor list — GPU kernels (Morton/AABB/block-pair-find/masks); provides mapping only |
| `mdpy/core/topology.py` | Molecular topology (particles/bonds/angles/dihedrals/impropers), `join()` → compact arrays |
| `mdpy/core/parameter_table.py` | ParameterTable with per-type and per-atom parameter dicts |
| `mdpy/force/force_term.py` | `ForceTerm` base class — `compute(gpu_context, block_list)` |
| `mdpy/force/bonded_force.py` | Single CuPy RawKernel (bond/angle/dihedral/improper); owns indices, remaps via `d_remap` |
| `mdpy/force/nonbonded_force.py` | CuPy RawKernel (self/cross block pair) + expression transpiler; owns sorted posq, params |
| `mdpy/force/force_group.py` | Homogeneous force composition; fuses nonbonded expressions into one kernel via `+` |
| `mdpy/force/primitives.py` | `param`/`scalar` markers + `distance`/`angle`/`dihedral` geometry helpers |
| `mdpy/force/ad_engine.py` | `ForwardADEngine` — forward-mode AD over a tape for CUDA gradient generation |
| `mdpy/force/bonded_transpiler.py` | `@bonded_expression(body=N)` → CUDA fragment; `HELPER_REGISTRY` forward+force templates |
| `mdpy/force/nonbonded_transpiler.py` | `@nonbonded_expression` → `energy_cuda`/`dEdr_cuda`; `__add__` fuses expressions |
| `mdpy/force/pme_reciprocal_force.py` | PME reciprocal space — the only non-expression `ForceTerm` subclass |
| `mdpy/force/factories/charmm.py` | PSF+topology+ParameterTable → wired force terms (`create_charmm_forces`) |
| `mdpy/force/expressions/lennard_jones.py` | LJ expression (decorate-then-override with closed-form gradient) |
| `mdpy/force/expressions/coulomb.py` | Coulomb expression (uses file-local derived `COULOMB_CONST`) |
| `mdpy/force/expressions/screened_coulomb.py` | Direct-space Coulomb (rational-minimax `erfc` override, `__MDPY_COULOMB__` placeholder) |
| `mdpy/integrator/verlet.py` | `@cuda.jit` Verlet integrator (`step(system)`) |
| `mdpy/integrator/langevin.py` | `@cuda.jit` Langevin BAOAB (`step(system)`, LCG PRNG) |
| `mdpy/io/` | File parsers (PSF/PDB/CHARMM toppar) |
| `benchmark/benchmark_1m9z.py` | 1M9Z performance benchmark with per-kernel GPU timing |
| `benchmark/validate_ion_openmm.py` | Ion-box OpenMM per-term validation (small pairwise system) |
| `benchmark/profile_openmm_nl.py` | OpenMM neighbor list nsys profiling workload |

## Development Notes

- All arrays default to `env.NUMPY_FLOAT` (float32) / `env.NUMPY_INT` (int32)
- GPU kernel strategy: both `cupy.RawKernel` and `numba.cuda.jit` are valid; choose based on kernel needs
- `Topology.join()` must be called before creating a System (`System.__init__` calls it automatically)
- Unit-dependent constants are **derived**, never hardcoded, and defined **file-locally** in each module that uses them (scope = that file): `COULOMB_CONST = 1/(4π·EPSILON0)` ≈ 0.13893557 (in `expressions/coulomb.py`/`nb14.py`/`screened_coulomb.py`/`pme_reciprocal_force.py`) and `BOLTZMANN = KB → default_energy_unit/K` ≈ 8.31446e-7 (in `integrator/langevin.py`/`utils/velocity.py`). Base constants `EPSILON0`/`KB` live in `mdpy/unit`. Values are in the internal energy unit `dalton·Å²/fs²` (≈ 9999.93 kJ/mol) — **not** kcal/mol
- The expression transpiler (nonbonded + bonded) resolves named module-level constants from a decorated function's `__globals__` and inlines them as CUDA float literals; hand-written CUDA strings (PME, screened_coulomb override) inject the value via the `__MDPY_COULOMB__` placeholder
- Block list rebuild: triggered when max atom displacement > skin/2; runs fully on GPU (Morton sort + block form + AABB + block-pair find + mask construction + block-pair classification)
- Expression transpiler: converts Python AST to CUDA C for CuPy RawKernel (used by both bonded and nonbonded forces)

## Known CPU Bottlenecks & Tech Debt

### P1 — Scalar Reads in Hot Path (acceptable, but could be cleaner)

Scalar reads for kernel launch sizing and control flow. These are acceptable per §4 but are noted as targets for future device-side control flow:

| Location | Scalar read | Purpose |
|----------|------------|---------|
| `block_list.py:973` | `int(d_num_blocks[0])`, `int(d_total_padded[0])` | Block count / padded size for array sizing |
| `block_list.py:1089` | `int(self._d_counters[0])` | Block-pair count for array sizing |
| `block_list.py:1284-1285` | `int(self._d_classify_excl_counter[0])`, `int(self._d_classify_main_counter[0])` | Exclusion/main block-pair counts |
| `block_list.py:1334` | `int(self.d_rebuild_flag[0])` | Rebuild decision (sync-interval checkpoint) |

### P2 — Dead Code to Remove

| Location | What to delete |
|----------|----------------|
| `gpu_context.py` | Any remaining `_HAS_CUPY` / `_HAS_GPU` / `_use_gpu` if present — replace with unconditional `import cupy as cp` |
| `block_list.py` | Any remaining `_HAS_CUPY` / `_HAS_GPU` if present — replace with unconditional `import cupy as cp` |
| `block_list.py` | `_rebuild_cpu()`, `_cut_blocks()`, `_compute_block_aabbs()`, `_find_tiles()`, `_morton_encode()`, `_aabb_min_image_dist_sq()`, `_to_device()` — CPU-only fallbacks if still present |
| `block_list.py` | Pure Python fallbacks for `_build_atom_to_block_slot` / `_build_masks_numba` (`else` branch) if still present |
| `environment.py` | `set_platform()`, `'CPU'` option, `supported_platforms` — platform is always CUDA |

> Note: the previous P2 item "`nonbonded_force.py:672` `getDeviceProperties` queried every compute call" is resolved — device properties are now cached once in `_lazy_compile()` (`nonbonded_force.py:502`).

### P3 — Performance Optimizations

| Location | Problem | Fix |
|----------|---------|-----|
| `block_list.py` block-pair find | block-pair buffers are overallocated (max-block-pairs heuristic) | Dynamic allocation via two-pass (count then fill) |
| `nonbonded_force.py` | `_gather_per_particle` re-gathers sorted per-particle arrays every rebuild even if unchanged | Skip when block list hasn't rebuilt |

## Physical Constants (internal units: Å / dalton / fs / e / K)

The internal energy unit is the derived natural unit `dalton·Å²/fs²` (≈ 9999.93 kJ/mol ≈ 2390 kcal/mol) — **not** kcal/mol and **not** kJ/mol.

Base physical constants (`EPSILON0`, `KB`, `NA`) live in `mdpy/unit/__init__.py`. The unit-dependent working constants below are **derived** from those bases and defined **file-locally** in each module that uses them (scope = that file), never hardcoded as magic numbers:

| Constant | Value (internal unit) | Derivation (file-local) | Defined in |
|----------|-------|--------|-----------|
| Coulomb constant (1/4πε₀) | ≈ 0.13893557 | `1/(4π·EPSILON0.value)`, dimension `dalton·Å²·Å/(fs²·e²)` | `expressions/coulomb.py`, `expressions/nb14.py`, `expressions/screened_coulomb.py`, `pme_reciprocal_force.py` |
| Boltzmann constant | ≈ 8.3144626e-7 | `KB.convert_to(default_energy_unit/kelvin).value` | `integrator/langevin.py`, `utils/velocity.py` |
| Conversion: OpenMM energy → mdpy internal | `× 1e-4` | 1 kJ/mol = 1e-4 internal unit | (test/reference generation) |
| Conversion: OpenMM force → mdpy internal | `× 1e-5` | 1 kJ/(mol·nm) = 1e-5 internal force unit | (test/reference generation) |
| Conversion: OpenMM position → mdpy | `× 10.0` | nm → Å | (test/reference generation) |

`dump_energy()` / `dump_forces()` return values in this internal unit. Converting to kJ/mol or any other unit is the caller's responsibility via `mdpy.unit.Quantity(...).convert_to(...)`.
