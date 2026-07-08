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

> **Exception**: one-off diagnostic/comparison scripts belong in `/tmp/opencode/`, **never** in `benchmark/` or any project directory. If the result is worth keeping permanently, ask before promoting it into the repo.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For mdpy specifically, "verified" usually means `conda run -n md_analysis pytest mdpy/test/ -sv` passes, plus `benchmark/benchmark_1m9z.py` runs clean after force/block-list changes.

### 5. Source Code First

**Every claim must be backed by source code evidence — not memory, not documentation, not convention.**

- When making a claim about how code works, cite the specific file and line number.
- Do NOT trust documentation (including this AGENTS.md) without verifying against actual source. Documentation can stale; source code is the ground truth.
- If uncertain about an API signature, class name, or method existence: read the file. Do not guess from memory.
- When analyzing a bug, trace the call path in source — do not hypothesize based on the function name.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Environment

- Always use `conda run -n md_analysis` (or `conda activate md_analysis`)
- Python 3.14 / numba 0.64 / numpy 2.4 / cupy 14.0 / openmm 8.4
- Dev install: `conda develop .` (editable install, source changes take effect immediately)

## GPU-Only Design Philosophy

mdpy is a GPU-native MD engine. Three hard rules:

### 1. Data Stays on GPU

All simulation data (positions, velocities, forces, energies, block list) resides in GPU memory for the entire simulation. CPU touches data only at two points:

- **Input**: loading structure files (PSF/PDB/PRM) → one-time CPU→GPU upload via `State.set_*()` methods
- **Output**: calling `dump_state()` or `dump_energy()` → on-demand GPU→CPU download

Specifically:
- `compute_forces()` accumulates forces on GPU via `atomicAdd` — the integrator reads `d_forces` directly on GPU
- `check_rebuild()` must NOT download all positions — uses a GPU reduction kernel that writes to a device flag
- Energy accumulation happens on GPU via `d_energy_accumulator` — no readback during computation
- PBC wrapping runs on GPU via `pbc_wrap_kernel`

### 2. Hard Dependencies, No CPU Path

`cupy` and `numba.cuda` are hard requirements. Import them unconditionally at module top level. There is no CPU execution path — no probing, no fallback.

- Do NOT write `try: import cupy` / `except ImportError` — let it crash on missing CUDA
- Do NOT use `_HAS_CUPY` / `_HAS_GPU` / `_use_gpu` runtime probes or `if self._use_gpu` branches
- No CPU fallback methods (`_rebuild_cpu()`) or pure-Python fallbacks for numba/cupy kernels
- `precision.py` exposes only dtype selection (`FLOAT`/`INT`); there is no platform field and no `set_platform` — platform is always CUDA

### 3. No GPU→CPU Transfers in Hot Path

The following functions — and every function they call directly or indirectly — must NOT transfer **bulk data** from GPU to CPU:

- `update_neighbor_list()`, `compute_forces()`, `integrator.step(system)`, `apply_constraints(time_step)`
- `minimizer.step(system)` inner loop

**Forbidden**: `cp.asnumpy()`, `array.get()`, GPU reductions followed by CPU readback, debug readback.

**Allowed**: Reading a single-element device array (e.g. `int(d_counter[0])`) for kernel launch parameters.

**All GPU→CPU bulk transfers must go through explicit output APIs:**

| API | Returns | GPU transfers |
|-----|---------|---------------|
| `System.set_positions(positions)` | — | CPU→GPU upload of `(N,3)` array |
| `System.set_velocities(velocities)` | — | CPU→GPU upload of `(N,3)` array |
| `System.dump_state()` | `(positions, velocities)` as numpy arrays, **PDB order** | download |
| `System.dump_forces()` | `forces` as `(N,3)` numpy array, **PDB order** | download |
| `System.dump_energy()` | `dict[str, float]` of per-term energies | `cp.asnumpy(d_energy_accumulator)` |

There are no `potential_energy` or `energies` properties on System. Energy is not "queryable state" — it is "output you explicitly request". Callers who need state during a loop collect it at explicit checkpoints (see **User-Facing Simulation Pattern** below).

## Architecture

```
PSFParser + PDBParser + CharmmTopparParser
  → create_parameter_table()
  → System(topology, state=State(num_particles))
     → State (pure PDB-order GPU storage: d_positions*, d_velocities*, d_forces*, d_masses*, d_charges*, d_type_indices)
     → BlockList (spatial sort + block-indexed masks + d_sorted_posq / d_sorted_type_indices buffers)
     → Topology (particles/bonds/angles/dihedrals/impropers + lazy atom-indexed exclusion state)
     → ForceTerms: BondedForce + NonbondedForce + PMEReciprocalForce (each owns PDB-order params)
     → Integrator: Verlet / Langevin BAOAB
  → GPU-only simulation loop → dump_state() / dump_energy() only when output needed
```

### Core Contracts

| File | Contract |
|------|----------|
| `mdpy/core/state.py` | Primary arrays in PDB order: `d_positions*`, `d_velocities*`, `d_forces*`, `d_masses`, `d_charges`, `d_type_indices`. PBC wrap, `zero_forces()`. **No sorted buffers. No neighbor lists. No force computation. No exclusion logic.** |
| `mdpy/core/block_list.py` | Spatial sort + block-indexed masks. Owns sorted gather buffers: `d_sorted_posq` (every step), `d_sorted_type_indices` (rebuild). Reads Topology CSR to build block-indexed exclusion masks. **Never permutes State's primary arrays — those stay in PDB order.** |
| `mdpy/core/topology.py` | Bonds/angles/dihedrals/impropers + lazy GPU exclusion state (`exclusion_pairs`, `exclusion_csr`, `exclusion_reverse_csr`). **Built once, cached, reused across spatial rebuilds.** |
| `mdpy/system.py` | Pipeline driver: `update_neighbor_list()` → `compute_forces()` → integrator → `apply_constraints()`. Output via `dump_*()`. |

### Per-step & Rebuild Flow

**`System.compute_forces()` every step:**
1. `state.zero_forces()`
2. `block_list.refresh_sorted_posq(state)` — gather PDB-order positions+charges into block-ordered SoA
3. `term.compute(state, block_list, compute_energy=False)` for each force term

**Block list rebuild** (triggered by displacement > skin/2):
1. `block_list.rebuild(topology, state, force=True)` — Morton sort → cell-aligned blocks
2. `state.wrap_positions_with_prev_correction()`
3. `block_list.capture_snapshot(state)`
4. `block_list.build_block_pairs(topology, state)` — neighbor pair maps + block-indexed exclusion masks
5. `block_list.refresh_sorted_type_indices(state)`

## Force System: Expressions, Not Subclasses

**Hard rule: do NOT write a new `ForceTerm` subclass to add a force.** mdpy has exactly two force engines — `BondedForce` and `NonbondedForce` — plus the `PMEReciprocalForce` exception. A new interaction is a Python *expression function* fed to one of those engines.

| You want | What to write | Engine |
|----------|---------------|--------|
| Bonded term (bond/angle/dihed/improper, 2-4 atoms) | `@bonded_expression(body=N)` function | `BondedForce(expr)` |
| Pairwise nonbonded term | `@nonbonded_expression` function | `NonbondedForce(expr, cutoff)` |
| Combine two nonbonded terms into one kernel | `expr1 + expr2`, or `force1 + force2` | Fuses into one block-pair kernel |
| Reciprocal-space PME (FFT/grid) | — | `PMEReciprocalForce` (only non-expression subclass) |

### Expression signatures

The transpiler classifies function parameters by their defaults:

- **Positions**: first `body` args (`body=2` → `pos1, pos2`; `body=3` → `p1, p2, p3`). Nonbonded body is always 2.
- **Per-term scalar param**: `name=param` or `name=<number>` (e.g. `k`, `r0`). Passed via `BondedForce.add(indices, **params)` or `NonbondedForce.set_pair_parameter`.
- **Compile-time scalar**: `name=scalar`. One value per kernel; passed via `set_scalar`.
- **Per-particle property**: bare arg name with no default. Trailing digits stripped to find base: `charge1`/`charge2` → base `charge`. i-props = no digit or `...1`, j-props = `...2`.

Import markers from `mdpy.force.markers`, geometry helpers from `mdpy.force.expressions.geometry`:

```python
from mdpy.force.markers import param, scalar
from mdpy.force.expressions.geometry import distance, angle, dihedral

@bonded_expression(body=2)
def harmonic_bond(p1, p2, k=param, r0=param):
    return 0.5 * k * (distance(p1, p2) - r0) ** 2
```

Module-level named constants (e.g. `COULOMB_CONST`) are resolved from the decorated function's `__globals__` and inlined as CUDA float literals.

### Transpiler pipeline

1. Python AST → `TapeEntry` tape + CUDA forward-evaluation lines
2. `ForwardADEngine` forward-mode AD: bonded uses `HELPER_REGISTRY` templates, nonbonded outputs `energy_cuda` + `radial_force_cuda`

### Decorate-then-override escape hatch

Overwrite compiled output after decoration for hand-tuned gradients. See `expressions/lennard_jones.py` and `expressions/screened_coulomb.py` for examples:

```python
@nonbonded_expression
def my_expr(pos1, pos2, ...):
    ...  # auto-compiled; output discarded

my_expr.energy_cuda = '...hand-written CUDA...'
my_expr.radial_force_cuda = '_my_force_var'
my_expr.grad_cuda = None
my_expr._local_vars = {...}  # for fusion renaming
```

Hand-written CUDA injects Coulomb constant via `__MDPY_COULOMB__` placeholder.

### Composition with `+`

- Expressions fuse: `expr1 + expr2` merges into one kernel
- `ForceGroup` requires homogeneous types — no mixing BondedForce and NonbondedForce

## User-Facing Simulation Pattern

Canonical wiring (see `benchmark/benchmark_1m9z.py`):

```python
from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.integrator.langevin import LangevinBAOABIntegrator
from mdpy.core.state import State
from mdpy.system import System
from mdpy.constraint.constraint_scheme import create_constraints
from mdpy.utils import generate_velocity_from_temperature

topology = psf.topology
parameter_table = create_parameter_table(topology, toppar)

state = State(topology.num_particles)
state.set_pbc(pbc_matrix)
state.set_positions(pdb.positions)
state.set_charges(psf.charges)
state.set_masses(psf.masses)
state.set_type_indices(psf.particle_type_indices)

system = System(topology, state=state)

forces = create_charmm_forces(topology, parameter_table, pbc_matrix, cutoff=12.0)
# returns {'bonded': ForceGroup, 'nonbonded': NonbondedForce, 'pme': PMEReciprocalForce}

system.add_force_term(forces["bonded"])
system.add_force_term(forces["nonbonded"])
system.add_force_term(forces["pme"])

constraints = create_constraints(topology, parameter_table, scheme="h-bonds")
for c in constraints:
    system.add_constraint(c)

system.set_velocities(generate_velocity_from_temperature(masses, temperature))

integrator = LangevinBAOABIntegrator(time_step_fs, temperature, friction)

for i in range(n_steps):
    system.update_neighbor_list(sync_interval=10)
    system.compute_forces()
    integrator.step(system)
    system.apply_constraints(time_step_fs)
    if i % checkpoint == 0:
        pos, vel = system.dump_state()
        energies = system.dump_energy()
```

Key contracts:
- `System(topology, state=None)` — `state` arg is optional; if omitted, System creates an empty State and seeds masses/charges/types from topology (transitional).
- PBC set via `system.set_pbc(pbc_matrix)` or directly `state.set_pbc(pbc_matrix)`.
- Positions and velocities must be set before `compute_forces()` / `update_neighbor_list()`.
- The per-step order is fixed: neighbor list → forces → integrate → constraints.

## GPU Kernel Strategy

Both `cupy.RawKernel` and `numba.cuda.jit` are acceptable:

- **`cupy.RawKernel`**: CUDA C strings. Used for force kernels (expression transpiler), block list kernels, PBC wrapping. Prefer for shared memory / warp-level control.
- **`numba.cuda.jit`**: Python-to-PTX. Used for integrators. Prefer for readability.

No `.cu` files. All kernels live in Python strings or numba-jitted Python functions.

## Naming Convention

No abbreviations. Readability first:

- `position`, `force`, `velocity`, `cutoff_radius`, `temperature`
- All quantity/count variables must use the `num_` prefix: `num_particles`, `num_blocks`, `num_cells`, `num_pairs` — not `n_particles`, `count_particles`, `particle_count`
- Exception: widely recognized physics abbreviations (`pbc`, `lj`, `pme`, `rmsd`, `rdf`)

## Git Workflow

After completing each task, commit changes using this workflow to ensure only YOUR modifications are committed:

```bash
# Step 1: At task start, record pre-existing modified files (skip if already recorded)
git diff --name-only > /tmp/mdpy_pre_task_dirty.txt

# Step 2: After completing the task, identify files changed during this session
comm -13 <(sort /tmp/mdpy_pre_task_dirty.txt) <(git diff --name-only | sort) > /tmp/mdpy_my_changes.txt

# If nothing changed, skip commit.
# If only new (untracked) files: git ls-files --others --exclude-standard

# Step 3: For non-trivial changes (core files, new files, refactors),
# dispatch the mdpy-code-reviewer subagent on each changed file.
# Address Critical and High findings. Skip for typo/comment-only changes.

# Step 4: Stage only the files YOU changed
xargs git add < /tmp/mdpy_my_changes.txt

# Step 5: Commit with a descriptive message
git commit -m "Brief description of what was done"

# Step 6: Push — rebase if remote has moved
git pull --rebase && git push
```

**Rules:**
- Never `git add -A` or `git add .` — only stage files identified as yours
- If `git pull --rebase` has conflicts, resolve them before pushing
- Use concise, descriptive commit messages (present tense, e.g. "Fix LJ cutoff edge case")
- Verify with `git diff --cached --stat` before committing

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

# Performance benchmark (1M9Z, 95,567 atoms)
CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/benchmark_1m9z.py

# Ion-box OpenMM validation (small pairwise system)
CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/benchmark_ion_openmm.py
```

- Test framework: pytest
- Test data directory: `mdpy/test/data/`
- OpenMM reference values: regenerate with `conda run -n md_analysis python mdpy/test/generate_reference.py`

**Targeted testing:** when making changes, identify and run only the related test file — do NOT run the full suite every time:

| Modified area | Test to run |
|--------------|-------------|
| `mdpy/core/state.py` | `pytest mdpy/test/test_state.py -sv` |
| `mdpy/core/block_list.py` | `pytest mdpy/test/test_block_list.py -sv` |
| `mdpy/core/topology.py` | `pytest mdpy/test/test_topology.py -sv` |
| `mdpy/system.py` | `pytest mdpy/test/test_system.py -sv` |
| `mdpy/force/bonded_force.py` | `pytest mdpy/test/test_bonded_force.py -sv` |
| `mdpy/force/nonbonded_force.py` | `pytest mdpy/test/test_nonbonded_force.py -sv` |
| Force expression changes | `pytest mdpy/test/ -sv -k "bonded or nonbonded"` |
| OpenMM validation | `pytest mdpy/test/test_openmm_validation.py -sv` |
| Major refactor / final verification | `pytest mdpy/test/ -sv` (full suite) |

### OpenMM Error Tolerances

| Force term | Energy relative error | Force absolute error | Reason |
|------------|----------------------|---------------------|--------|
| bond | < 1e-6 | < 1e-4 | Analytic formula |
| angle | < 1e-6 | < 1e-4 | Analytic formula |
| dihedral | < 1e-6 | < 1e-4 | Analytic formula |
| improper | < 1e-6 | < 1e-4 | Analytic formula |
| nonbonded (LJ) | < 1e-3 | < 1e-2 | Cutoff/switching differences |
| electrostatic | < 1e-3 | < 1e-2 | Direct Coulomb comparison |

## GPU Kernel Profiling

| Tool | Version | Path |
|------|---------|------|
| Nsight Compute (`ncu`) | 2025.3.1 | `/home/ubuntu/Programs/cuda/13.0/bin/ncu` |
| Nsight Systems (`nsys`) | 2025.3.2 | `/home/ubuntu/Programs/cuda/13.0/bin/nsys` |
| `nvtx` Python package | installed | `conda run -n md_analysis pip show nvtx` |

### Output Directories & Naming

- **nsys** output to `benchmark/nsys/`, **ncu** output to `benchmark/ncu/`
- Existing files in each directory show the naming convention — follow it:

**nsys**: `YYYY-MM-DD-NNN-{engine}-{system}-{description}` (e.g. `2026-07-07-001-mdpy-1m9z-postrefactor`)
- `NNN`: 3-digit daily sequence, starts at 001 each day. Check `ls benchmark/nsys/` for the next available number.
- `engine`: `mdpy` or `openmm`
- `system`: `1m9z`, `ion`, `stmv`
- `description`: short tag (`postrefactor`, `slot-optimized`, `block-ordered-types`, etc.)

**ncu**: `YYYY-MM-DD-NNN-{system}-{description}` (e.g. `2026-07-07-001-1m9z-exclusion-block-pair`)
- NNN: same rules as nsys. Check `ls benchmark/ncu/`.
- Engine defaults to `mdpy` (omit); put `openmm` in description when comparing.
- ncu `.ncu-rep` is the primary output file.

**nsys timeline:**
```bash
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/nsys profile \
  --trace=cuda,nvtx,osrt --sample=cpu \
  --output=benchmark/nsys/YYYY-MM-DD-NNN-mdpy-1m9z-xxx \
  conda run -n md_analysis python benchmark/benchmark_1m9z.py
```

**ncu deep dive on a specific kernel:**
```bash
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/ncu \
  --set full --launch-skip 10 --launch-count 5 \
  -k "regex:kernel_name_pattern" \
  -o benchmark/ncu/YYYY-MM-DD-NNN-1m9z-xxx \
  conda run -n md_analysis python benchmark/benchmark_1m9z.py
```

**NVTX markers** (for Python-side region labeling): `cp.cuda.nvtx.RangePush("name")` / `cp.cuda.nvtx.RangePop()` for CuPy, or `nvtx.range_push("name")` / `nvtx.range_pop()` from the `nvtx` pip package.

### Benchmark Parameters

**Temporary NUM_BLOCKS reduction:** profiling is slow; reduce blocks before running:
- `benchmark/benchmark_1m9z.py:32`: change `NUM_BLOCKS = 5` to `NUM_BLOCKS = 2` before profiling
- Restore to `5` immediately after profiling is done
- Same principle applies to any other benchmark script — use the minimum viable block count

**Notes:**
- Always set `CUDA_VISIBLE_DEVICES=0` (6-GPU environment)
- ncu is ~10-100× slower; use `-k` filter and `--launch-count` to limit iterations
- OpenMM profiling uses `Platform.getPlatformByName('CUDA')` with `PME` + 12Å cutoff

## Physical Constants

Internal units: Å / dalton / fs / e / K. The energy unit is `dalton·Å²/fs²` (≈ 9999.93 kJ/mol ≈ 2390 kcal/mol).

Base constants (`EPSILON0`, `KB`, `NA`) live in `mdpy/unit/__init__.py`. All derived constants are defined **file-locally** in each module that uses them — never hardcoded:

| Constant | Value | File-local definition | Defined in |
|----------|-------|----------------------|-----------|
| Coulomb constant (1/4πε₀) | ≈ 0.13893557 | `1/(4π·EPSILON0.value)` | `expressions/coulomb.py`, `screened_coulomb.py`, `pme_reciprocal_force.py` |
| Boltzmann constant | ≈ 8.31446e-7 | `KB.convert_to(energy_unit/kelvin).value` | `integrator/langevin.py`, `utils/velocity.py` |
| OpenMM energy → mdpy | × 1e-4 | 1 kJ/mol = 1e-4 internal unit | test/reference generation |
| OpenMM force → mdpy | × 1e-5 | 1 kJ/(mol·nm) = 1e-5 internal force | test/reference generation |
| OpenMM position → mdpy | × 10.0 | nm → Å | test/reference generation |

`dump_energy()` / `dump_forces()` return internal units. Conversion is the caller's responsibility.
