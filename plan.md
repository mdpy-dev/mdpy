# OpenMM Diagonal Loop + Static Dispatch Nonbonded Kernel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the naive per-pair `atomicAdd` nonbonded kernel with OpenMM-style diagonal loop + warp shuffle rotation, reducing atomicAdd from 6.0/pair to ~0.19/pair.

**Architecture:** One warp (32 threads) processes one tile (32x32 atom pairs). Each thread holds one i-atom, iterating 32 steps with rotate-right-1 shuffle to visit all j-atoms. i-force accumulates in register; j-force rotates with j-atom data and returns to origin after 32 steps. Static dispatch: each warp independently computes its tile range via integer division. Self-tile (diagonal) uses broadcast mode with `j != tgx` + `0.5f * energy_val`. Cross-tile uses rotation mode.

**Tech Stack:** CuPy RawKernel (CUDA C), `__shfl_sync`, `__shfl_down_sync`, Python AST transpiler (existing, unchanged).

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `mdpy/force/nonbonded_force.py` | **Major modify** | New code generators + new kernel assembly functions + updated `NonbondedExpression` methods + updated `NonbondedForce.compute()` |
| `mdpy/test/test_nonbonded_shuffle.py` | **Create** | Tests for new kernel correctness |
| `mdpy/test/test_nonbonded_expression.py` | **Modify** | Update kernel-source assertions for new structure |
| `mdpy/core/gpu_context.py` | **No change** | Energy uses existing `d_energy` (warp-level atomicAdd, no new buffers) |
| `mdpy/test/test_openmm_validation.py` | **No change** | Acceptance gate (6PO6 OpenMM validation) |

---

### Task 1: New Code Generators (5 Functions)

Create helper functions that generate CUDA C fragments for the shuffle-based kernels. These produce code for parameter loading, warp shuffle, and 1-4 parameter select/restore.

**Files:**
- Modify: `mdpy/force/nonbonded_force.py` (add 5 functions after line 298)
- Create: `mdpy/test/test_nonbonded_shuffle.py`

- [ ] **Step 1: Write the failing tests**

Create `mdpy/test/test_nonbonded_shuffle.py`:

```python
import pytest


class TestCodeGenerators:
    @pytest.fixture
    def params(self):
        return ['sigma', 'epsilon', 'charge']

    def test_param_load_i(self, params):
        from mdpy.force.nonbonded_force import _generate_param_load_i
        code = _generate_param_load_i(params)
        assert 'float sigma_i = 0.0f' in code
        assert 'sigma_i = sigma[gi]' in code
        assert 'sigma_i_14 = sigma_14[gi]' in code
        assert 'epsilon_i = epsilon[gi]' in code
        assert 'charge_i_14 = charge_14[gi]' in code

    def test_param_load_j_init(self, params):
        from mdpy.force.nonbonded_force import _generate_param_load_j_init
        code = _generate_param_load_j_init(params)
        assert 'float sigma_j = 0.0f' in code
        assert 'sigma_j = sigma[gj_init]' in code
        assert 'sigma_j_14 = sigma_14[gj_init]' in code

    def test_shuffle_warp_data(self, params):
        from mdpy.force.nonbonded_force import _generate_shuffle_warp_data
        code = _generate_shuffle_warp_data(params)
        assert '__shfl_sync(0xffffffff, shfl_px, (tgx + 1) & 31)' in code
        assert '__shfl_sync(0xffffffff, shfl_fx, (tgx + 1) & 31)' in code
        assert '__shfl_sync(0xffffffff, sigma_j, (tgx + 1) & 31)' in code
        assert '__shfl_sync(0xffffffff, sigma_j_14, (tgx + 1) & 31)' in code
        assert '__shfl_sync(0xffffffff, charge_j_14, (tgx + 1) & 31)' in code

    def test_param_select(self, params):
        from mdpy.force.nonbonded_force import _generate_param_select
        code = _generate_param_select(params)
        assert 'sigma_i_saved = sigma_i' in code
        assert 'if (is_14) sigma_i = sigma_i_14' in code
        assert 'sigma_j_saved = sigma_j' in code
        assert 'if (is_14) sigma_j = sigma_j_14' in code
        assert 'if (is_14) epsilon_i = epsilon_i_14' in code

    def test_param_restore(self, params):
        from mdpy.force.nonbonded_force import _generate_param_restore
        code = _generate_param_restore(params)
        assert 'if (is_14) sigma_i = sigma_i_saved' in code
        assert 'if (is_14) sigma_j = sigma_j_saved' in code
        assert 'if (is_14) epsilon_i = epsilon_i_saved' in code
        assert 'if (is_14) charge_j = charge_j_saved' in code
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n md_analysis pytest mdpy/test/test_nonbonded_shuffle.py -sv -k "CodeGenerators"`
Expected: FAIL (`ImportError: cannot import name '_generate_param_load_i'`)

- [ ] **Step 3: Implement the 5 code generator functions**

In `mdpy/force/nonbonded_force.py`, add after `_generate_pre_fetch` (after line 298):

```python
def _generate_param_load_i(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_i = 0.0f, {name}_i_14 = 0.0f;')
        lines.append(f'if (gi >= 0) {{ {name}_i = {name}[gi]; {name}_i_14 = {name}_14[gi]; }}')
    return '\n        '.join(lines)


def _generate_param_load_j_init(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_j = 0.0f, {name}_j_14 = 0.0f;')
        lines.append(f'if (gj_init >= 0) {{ {name}_j = {name}[gj_init]; {name}_j_14 = {name}_14[gj_init]; }}')
    return '\n        '.join(lines)


def _generate_shuffle_warp_data(parameter_names):
    lines = [
        'shfl_px = __shfl_sync(0xffffffff, shfl_px, (tgx + 1) & 31);',
        'shfl_py = __shfl_sync(0xffffffff, shfl_py, (tgx + 1) & 31);',
        'shfl_pz = __shfl_sync(0xffffffff, shfl_pz, (tgx + 1) & 31);',
        'shfl_fx = __shfl_sync(0xffffffff, shfl_fx, (tgx + 1) & 31);',
        'shfl_fy = __shfl_sync(0xffffffff, shfl_fy, (tgx + 1) & 31);',
        'shfl_fz = __shfl_sync(0xffffffff, shfl_fz, (tgx + 1) & 31);',
    ]
    for name in parameter_names:
        lines.append(
            f'{name}_j = __shfl_sync(0xffffffff, {name}_j, (tgx + 1) & 31);'
        )
        lines.append(
            f'{name}_j_14 = __shfl_sync(0xffffffff, {name}_j_14, (tgx + 1) & 31);'
        )
    return '\n        '.join(lines)


def _generate_param_select(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_i_saved = {name}_i;')
        lines.append(f'if (is_14) {name}_i = {name}_i_14;')
        lines.append(f'float {name}_j_saved = {name}_j;')
        lines.append(f'if (is_14) {name}_j = {name}_j_14;')
    return '\n            '.join(lines)


def _generate_param_restore(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'if (is_14) {name}_i = {name}_i_saved;')
        lines.append(f'if (is_14) {name}_j = {name}_j_saved;')
    return '\n            '.join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n md_analysis pytest mdpy/test/test_nonbonded_shuffle.py -sv -k "CodeGenerators"`
Expected: ALL PASS

- [ ] **Step 5: Run full suite to verify no regression**

Run: `conda run -n md_analysis pytest mdpy/test/ -sv --co -q`
Expected: 122 tests collected, no import errors

- [ ] **Step 6: Commit**

```bash
git add mdpy/force/nonbonded_force.py mdpy/test/test_nonbonded_shuffle.py
git commit -m "feat: add shuffle-based kernel code generators for diagonal loop"
```

---

### Task 2: Cross-Tile Kernel v2

Build the full cross-tile kernel source using static dispatch + diagonal loop + shuffle rotation. Wire it into `NonbondedExpression.assemble_cross_tile_kernel`.

**Key design:**
- Static dispatch: each warp computes `pos = warp_id * num_cross / total_warps` (no dynamic counter, no `__syncthreads`)
- Exclusion mask pre-rotation: `(excl >> tgx) | (excl << (32 - tgx))` aligns bits to shuffle pattern
- Convention: bit set = excluded (matches `tile_list.py` mask encoding)
- Both i and j params get 1-4 treatment via select/restore
- Energy: warp-level `__shfl_down_sync` reduction, one `atomicAdd` per warp per tile

**Files:**
- Modify: `mdpy/force/nonbonded_force.py` — add `_assemble_cross_tile_kernel_v2`, update `NonbondedExpression.assemble_cross_tile_kernel`
- Modify: `mdpy/test/test_nonbonded_shuffle.py`

- [ ] **Step 1: Write the failing tests**

Append to `mdpy/test/test_nonbonded_shuffle.py`:

```python
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb


class TestCrossTileKernelV2:
    @pytest.fixture
    def combined_expr(self):
        return lennard_jones + coulomb

    def test_kernel_has_static_dispatch(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'total_warps' in source
        assert 'warp_id * num_cross / total_warps' in source

    def test_kernel_has_shfl_sync(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert '__shfl_sync' in source

    def test_kernel_has_rotate_right(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert '(tgx + 1) & 31' in source

    def test_kernel_has_forces_register(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'shfl_fx' in source
        assert 'force_x' in source

    def test_kernel_has_32_step_loop(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'j < 32' in source

    def test_kernel_no_dynamic_counter(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'tile_counter' not in source

    def test_kernel_no_shared_memory(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert '__shared__' not in source

    def test_kernel_has_exclusion_prerotate(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'excl >> tgx' in source
        assert 'excl << (32 - tgx)' in source

    def test_kernel_exclusion_convention(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert '(excl & 0x1) != 0' in source

    def test_kernel_has_atomicAdd_force(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'atomicAdd(&forces[gi * 3' in source
        assert 'atomicAdd(&forces[gj * 3' in source

    def test_kernel_has_warp_energy_reduce(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert '__shfl_down_sync' in source
        assert 'atomicAdd(energy_buffer, energy)' in source

    def test_kernel_has_param_select_both(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'sigma_i_saved' in source
        assert 'if (is_14) sigma_i = sigma_i_14' in source

    def test_kernel_valid_braces(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert source.count('{') == source.count('}')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n md_analysis pytest mdpy/test/test_nonbonded_shuffle.py -sv -k "CrossTileKernelV2"`
Expected: FAIL (old kernel uses `__shared__`, has `tile_counter`, no `__shfl_sync`)

- [ ] **Step 3: Implement `_assemble_cross_tile_kernel_v2`**

In `mdpy/force/nonbonded_force.py`, add after `_generate_param_restore`:

```python
def _assemble_cross_tile_kernel_v2(parameter_names, expression_fragment):
    param_decls = _generate_param_decls(parameter_names)
    param_load_i = _generate_param_load_i(parameter_names)
    param_load_j = _generate_param_load_j_init(parameter_names)
    shuffle_code = _generate_shuffle_warp_data(parameter_names)
    param_select = _generate_param_select(parameter_names)
    param_restore = _generate_param_restore(parameter_names)

    kernel = f'''extern "C" __global__
void cross_tile_kernel(
    const float* positions,
    float* forces,
    float* energy_buffer,
    const int* block_atoms,
    const int* cross_tiles_i,
    const int* cross_tiles_j,
    const float* cross_tiles_shift,
    const unsigned int* cross_exclusion_masks,
    const unsigned int* cross_scaling_masks,
    float cutoff_sq,
    int num_cross{param_decls}
) {{
    int total_warps = (blockDim.x * gridDim.x) / 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tgx = threadIdx.x & 31;

    int pos = warp_id * num_cross / total_warps;
    int end = (warp_id + 1) * num_cross / total_warps;

    float energy = 0.0f;

    for (; pos < end; pos++) {{
        int bi = cross_tiles_i[pos];
        int bj = cross_tiles_j[pos];
        float shift_x = cross_tiles_shift[pos * 3 + 0];
        float shift_y = cross_tiles_shift[pos * 3 + 1];
        float shift_z = cross_tiles_shift[pos * 3 + 2];

        int gi = block_atoms[bi * 32 + tgx];
        float px_i = 0.0f, py_i = 0.0f, pz_i = 0.0f;
        if (gi >= 0) {{
            px_i = positions[gi * 3 + 0];
            py_i = positions[gi * 3 + 1];
            pz_i = positions[gi * 3 + 2];
        }}
        {param_load_i}

        int gj_init = block_atoms[bj * 32 + tgx];
        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f;
        if (gj_init >= 0) {{
            shfl_px = positions[gj_init * 3 + 0] + shift_x;
            shfl_py = positions[gj_init * 3 + 1] + shift_y;
            shfl_pz = positions[gj_init * 3 + 2] + shift_z;
        }}
        {param_load_j}

        float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
        float shfl_fx = 0.0f, shfl_fy = 0.0f, shfl_fz = 0.0f;

        unsigned int excl = cross_exclusion_masks[pos * 32 + tgx];
        excl = (excl >> tgx) | (excl << (32 - tgx));
        unsigned int scale = cross_scaling_masks[pos * 32 + tgx];
        scale = (scale >> tgx) | (scale << (32 - tgx));

        for (int j = 0; j < 32; j++) {{
            float dx = shfl_px - px_i;
            float dy = shfl_py - py_i;
            float dz = shfl_pz - pz_i;
            float dist_sq = dx * dx + dy * dy + dz * dz;

            bool excluded = (excl & 0x1) != 0;
            bool is_14 = (scale & 0x1) != 0;

            if (!excluded && dist_sq <= cutoff_sq && dist_sq > 1.0e-12f && gi >= 0) {{
                float inv_dist = rsqrtf(dist_sq);
                float r = dist_sq * inv_dist;

                {param_select}

                {expression_fragment}

                float inv_dist_force = force_magnitude * inv_dist;
                float fx = dx * inv_dist_force;
                float fy = dy * inv_dist_force;
                float fz = dz * inv_dist_force;

                force_x -= fx;
                force_y -= fy;
                force_z -= fz;
                shfl_fx += fx;
                shfl_fy += fy;
                shfl_fz += fz;
                energy += energy_val;

                {param_restore}
            }}

            {shuffle_code}

            excl >>= 1;
            scale >>= 1;
        }}

        if (gi >= 0) {{
            atomicAdd(&forces[gi * 3 + 0], force_x);
            atomicAdd(&forces[gi * 3 + 1], force_y);
            atomicAdd(&forces[gi * 3 + 2], force_z);
        }}
        int gj = block_atoms[bj * 32 + tgx];
        if (gj >= 0) {{
            atomicAdd(&forces[gj * 3 + 0], shfl_fx);
            atomicAdd(&forces[gj * 3 + 1], shfl_fy);
            atomicAdd(&forces[gj * 3 + 2], shfl_fz);
        }}
    }}

    for (int offset = 16; offset > 0; offset >>= 1) {{
        energy += __shfl_down_sync(0xffffffff, energy, offset);
    }}
    if (tgx == 0) atomicAdd(energy_buffer, energy);
}}'''
    return kernel
```

- [ ] **Step 4: Update `NonbondedExpression.assemble_cross_tile_kernel` to call v2**

In `mdpy/force/nonbonded_force.py`, replace `assemble_cross_tile_kernel` method (lines 237-242):

```python
    def assemble_cross_tile_kernel(self):
        fragment = self.cuda_fragment
        if '_result_energy_1' not in fragment:
            fragment += '\nfloat energy_val = _result_energy;'
            fragment += '\nfloat force_magnitude = _result_force;'
        return _assemble_cross_tile_kernel_v2(self.parameter_names, fragment)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n md_analysis pytest mdpy/test/test_nonbonded_shuffle.py -sv -k "CrossTileKernelV2"`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add mdpy/force/nonbonded_force.py mdpy/test/test_nonbonded_shuffle.py
git commit -m "feat: implement cross-tile kernel with static dispatch + diagonal loop + shuffle"
```

---

### Task 3: Self-Tile Kernel v2 (Broadcast Mode)

Rewrite the self-tile kernel to use OpenMM's broadcast mode. Same block (x == y), so `__shfl_sync` broadcast from lane j instead of rotation. All j (with `j != tgx`) to avoid self-interaction. Each pair computed by both threads (i sees j, j sees i), so `0.5f * energy_val` for correct energy.

**Why not upper triangle (`j > tgx`)?** With `j > tgx`, each pair is computed once but only the i-force is accumulated — the j-force (Newton's 3rd law) is lost because there's no rotation mechanism to deliver it. Using all j lets each thread accumulate its own atom's force from every other atom, which is correct without explicit Newton's 3rd law application. Energy is halved because each pair is counted twice.

**Files:**
- Modify: `mdpy/force/nonbonded_force.py` — add `_assemble_self_tile_kernel_v2`, update `NonbondedExpression.assemble_self_tile_kernel`
- Modify: `mdpy/test/test_nonbonded_shuffle.py`

- [ ] **Step 1: Write the failing tests**

Append to `mdpy/test/test_nonbonded_shuffle.py`:

```python
class TestSelfTileKernelV2:
    @pytest.fixture
    def combined_expr(self):
        return lennard_jones + coulomb

    def test_self_kernel_has_32_step_loop(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'j < 32' in source

    def test_self_kernel_has_broadcast(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert '__shfl_sync(0xffffffff, px_i, j)' in source

    def test_self_kernel_no_naive_pair_loop(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'linear = tid * 2 + iter' not in source
        assert 'linear >= 496' not in source

    def test_self_kernel_no_upper_triangle(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'j > tgx' not in source

    def test_self_kernel_has_j_neq_tgx(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'j != tgx' in source

    def test_self_kernel_has_half_energy(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert '0.5f * energy_val' in source

    def test_self_kernel_no_shared_mem_positions(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'smem_pos' not in source

    def test_self_kernel_has_exclusion_shift(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'excl >>= 1' in source

    def test_self_kernel_has_only_i_force(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'force_x' in source
        lines = source.split('\n')
        shfl_force_lines = [l for l in lines if 'shfl_f' in l and '__shfl_sync' not in l]
        assert len(shfl_force_lines) == 0

    def test_self_kernel_has_warp_energy_reduce(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert '__shfl_down_sync' in source

    def test_self_kernel_has_param_broadcast(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert '__shfl_sync(0xffffffff, sigma_i, j)' in source
        assert '__shfl_sync(0xffffffff, sigma_i_14, j)' in source

    def test_self_kernel_valid_braces(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert source.count('{') == source.count('}')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n md_analysis pytest mdpy/test/test_nonbonded_shuffle.py -sv -k "SelfTileKernelV2"`
Expected: FAIL (old kernel uses `smem_pos`, `linear >= 496`, no `__shfl_sync`)

- [ ] **Step 3: Implement `_assemble_self_tile_kernel_v2`**

In `mdpy/force/nonbonded_force.py`, add after `_assemble_cross_tile_kernel_v2`:

```python
def _assemble_self_tile_kernel_v2(parameter_names, expression_fragment):
    param_decls = _generate_param_decls(parameter_names)
    param_load_i_lines = []
    param_load_i_14_lines = []
    broadcast_j_lines = []
    select_lines = []
    restore_lines = []
    for name in parameter_names:
        param_load_i_lines.append(f'float {name}_i = 0.0f, {name}_i_14 = 0.0f;')
        param_load_i_lines.append(f'if (gi >= 0) {{ {name}_i = {name}[gi]; {name}_i_14 = {name}_14[gi]; }}')
        broadcast_j_lines.append(
            f'float {name}_j = __shfl_sync(0xffffffff, {name}_i, j);'
        )
        broadcast_j_lines.append(
            f'float {name}_j_14 = __shfl_sync(0xffffffff, {name}_i_14, j);'
        )
        select_lines.append(f'float {name}_i_saved = {name}_i;')
        select_lines.append(f'if (is_14) {name}_i = {name}_i_14;')
        select_lines.append(f'float {name}_j_saved = {name}_j;')
        select_lines.append(f'if (is_14) {name}_j = {name}_j_14;')
        restore_lines.append(f'if (is_14) {name}_i = {name}_i_saved;')
        restore_lines.append(f'if (is_14) {name}_j = {name}_j_saved;')

    param_load_i = '\n    '.join(param_load_i_lines)
    broadcast_j = '\n            '.join(broadcast_j_lines)
    param_select = '\n            '.join(select_lines)
    param_restore = '\n            '.join(restore_lines)

    kernel = f'''extern "C" __global__
void self_tile_kernel(
    const float* positions,
    float* forces,
    float* energy_buffer,
    const int* block_atoms,
    const int* self_tile_indices,
    const unsigned int* self_exclusion_masks,
    const unsigned int* self_scaling_masks,
    float cutoff_sq{param_decls}
) {{
    int tile_idx = blockIdx.x;
    int tgx = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    int block_k = self_tile_indices[tile_idx];
    float energy = 0.0f;

    int gi = block_atoms[block_k * 32 + tgx];
    float px_i = 0.0f, py_i = 0.0f, pz_i = 0.0f;
    if (gi >= 0) {{
        px_i = positions[gi * 3 + 0];
        py_i = positions[gi * 3 + 1];
        pz_i = positions[gi * 3 + 2];
    }}
    {param_load_i}

    float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;

    unsigned int excl = self_exclusion_masks[tile_idx * 32 + tgx];
    unsigned int scale = self_scaling_masks[tile_idx * 32 + tgx];

    for (int j = 0; j < 32; j++) {{
        int gj = block_atoms[block_k * 32 + j];
        float px_j = __shfl_sync(0xffffffff, px_i, j);
        float py_j = __shfl_sync(0xffffffff, py_i, j);
        float pz_j = __shfl_sync(0xffffffff, pz_i, j);

        float dx = px_j - px_i;
        float dy = py_j - py_i;
        float dz = pz_j - pz_i;
        float dist_sq = dx * dx + dy * dy + dz * dz;

        bool excluded = (excl & 0x1) != 0;
        bool is_14 = (scale & 0x1) != 0;

        if (!excluded && dist_sq <= cutoff_sq && dist_sq > 1.0e-12f
            && gi >= 0 && gj >= 0 && j != tgx) {{
            float inv_dist = rsqrtf(dist_sq);
            float r = dist_sq * inv_dist;

            {broadcast_j}
            {param_select}

            {expression_fragment}

            float inv_dist_force = force_magnitude * inv_dist;
            force_x -= dx * inv_dist_force;
            force_y -= dy * inv_dist_force;
            force_z -= dz * inv_dist_force;
            energy += 0.5f * energy_val;

            {param_restore}
        }}

        excl >>= 1;
        scale >>= 1;
    }}

    if (gi >= 0) {{
        atomicAdd(&forces[gi * 3 + 0], force_x);
        atomicAdd(&forces[gi * 3 + 1], force_y);
        atomicAdd(&forces[gi * 3 + 2], force_z);
    }}

    for (int offset = 16; offset > 0; offset >>= 1) {{
        energy += __shfl_down_sync(0xffffffff, energy, offset);
    }}
    if (tgx == 0) atomicAdd(energy_buffer, energy);
}}'''
    return kernel
```

- [ ] **Step 4: Update `NonbondedExpression.assemble_self_tile_kernel`**

In `mdpy/force/nonbonded_force.py`, replace `assemble_self_tile_kernel` method (lines 230-235):

```python
    def assemble_self_tile_kernel(self):
        fragment = self.cuda_fragment
        if '_result_energy_1' not in fragment:
            fragment += '\nfloat energy_val = _result_energy;'
            fragment += '\nfloat force_magnitude = _result_force;'
        return _assemble_self_tile_kernel_v2(self.parameter_names, fragment)
```

- [ ] **Step 5: Run self-tile tests**

Run: `conda run -n md_analysis pytest mdpy/test/test_nonbonded_shuffle.py -sv -k "SelfTileKernelV2"`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add mdpy/force/nonbonded_force.py mdpy/test/test_nonbonded_shuffle.py
git commit -m "feat: implement self-tile kernel with broadcast mode and warp shuffle"
```

---

### Task 4: Update NonbondedForce.compute()

Remove `tile_counter` dynamic dispatch, update `compute()` to use static dispatch launch config, remove dead `_d_tile_counter` allocation.

**Files:**
- Modify: `mdpy/force/nonbonded_force.py:584-678` (`NonbondedForce` class)

- [ ] **Step 1: Write the failing test**

Append to `mdpy/test/test_nonbonded_shuffle.py`:

```python
class TestNonbondedForceCompute:
    def test_compute_no_tile_counter(self):
        from mdpy.force.nonbonded_force import NonbondedForce
        nf = NonbondedForce(lennard_jones + coulomb)
        assert not hasattr(nf, '_d_tile_counter') or nf._d_tile_counter is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n md_analysis pytest mdpy/test/test_nonbonded_shuffle.py -sv -k "test_compute_no_tile_counter"`
Expected: FAIL (`_d_tile_counter` is initialized in `__init__`)

- [ ] **Step 3: Update NonbondedForce class**

In `mdpy/force/nonbonded_force.py`, make these changes to the `NonbondedForce` class:

**3a. Remove `_d_tile_counter` from `__init__`** (line 597):

Replace:
```python
        self._d_tile_counter = None
```
With nothing (delete the line).

**3b. Remove `_d_tile_counter` allocation from `_ensure_compiled`** (line 628):

The line:
```python
        self._d_tile_counter = cp.zeros(2, dtype=cp.int32)
```
Delete this line.

**3c. Replace `compute` method** (lines 637-678):

```python
    def compute(self, gpu_context, tile_list=None):
        self._ensure_compiled()

        if tile_list is None:
            return 0.0

        if tile_list.num_self > 0:
            self_args = [
                gpu_context.d_positions,
                gpu_context.d_forces,
                gpu_context.d_energy,
                tile_list.d_block_atoms,
                tile_list.d_self_tile_indices,
                tile_list.d_self_exclusion_masks,
                tile_list.d_self_scaling_masks,
                np.float32(self._cutoff_sq),
            ] + self._param_args()
            self._self_kernel(
                (tile_list.num_self,), (256,),
                tuple(self_args),
            )

        if tile_list.num_cross > 0:
            cross_args = [
                gpu_context.d_positions,
                gpu_context.d_forces,
                gpu_context.d_energy,
                tile_list.d_block_atoms,
                tile_list.d_cross_tiles_i,
                tile_list.d_cross_tiles_j,
                tile_list.d_cross_tiles_shift,
                tile_list.d_cross_exclusion_masks,
                tile_list.d_cross_scaling_masks,
                np.float32(self._cutoff_sq),
                np.int32(tile_list.num_cross),
            ] + self._param_args()
            num_sm = self._num_sm
            cross_grid = 4 * num_sm
            self._cross_kernel(
                (cross_grid,), (256,),
                tuple(cross_args),
            )

        if tile_list.num_interactions == 0:
            return 0.0
        return None
```

Note: `return None` defers the energy readback to the caller (`system.py:43` does `float(self.gpu.d_energy[0])`). The self-tile kernel signature no longer includes `num_cross` (it doesn't use it).

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n md_analysis pytest mdpy/test/test_nonbonded_shuffle.py -sv -k "test_compute_no_tile_counter"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mdpy/force/nonbonded_force.py mdpy/test/test_nonbonded_shuffle.py
git commit -m "feat: update NonbondedForce.compute() for static dispatch, remove tile_counter"
```

---

### Task 5: Update Existing Nonbonded Expression Tests

The existing `test_nonbonded_expression.py` has assertions against the old kernel structure (shared memory, conditional param loading, flat pair loop). Update to match the new shuffle-based kernel.

**Files:**
- Modify: `mdpy/test/test_nonbonded_expression.py`

- [ ] **Step 1: Run existing tests to identify failures**

Run: `conda run -n md_analysis pytest mdpy/test/test_nonbonded_expression.py -sv`
Expected: Some tests fail. The failing tests are:

| Test | Reason |
|------|--------|
| `TestKernelAssembly::test_lj_kernel_structure` | Asserts `__shared__` in kernel (new kernel has none) |
| `TestCombinedKernelSource::test_combined_kernel_source` | Asserts `sigma_i = is_14 ? sigma_14[gi] : sigma[gi]` (new kernel loads both, selects later) |

- [ ] **Step 2: Fix `test_lj_kernel_structure`**

In `mdpy/test/test_nonbonded_expression.py`, replace the test at line 156:

```python
    def test_lj_kernel_structure(self):
        kernel = lennard_jones.assemble_cross_tile_kernel()
        assert 'extern "C" __global__' in kernel
        assert 'void cross_tile_kernel' in kernel
        assert '__shfl_sync' in kernel
        assert 'atomicAdd' in kernel
        assert 'rsqrtf' in kernel
        assert 'is_14' in kernel
        assert 'energy_val' in kernel
        assert 'force_magnitude' in kernel
```

- [ ] **Step 3: Fix `TestCombinedKernelSource.test_combined_kernel_source`**

In `mdpy/test/test_nonbonded_expression.py`, replace the test at line 277:

```python
    def test_combined_kernel_source(self):
        combined = lennard_jones + coulomb
        kernel = combined.assemble_cross_tile_kernel()

        assert 'extern "C" __global__' in kernel
        assert 'void cross_tile_kernel' in kernel

        assert 'const float* sigma' in kernel
        assert 'const float* sigma_14' in kernel
        assert 'const float* epsilon' in kernel
        assert 'const float* epsilon_14' in kernel
        assert 'const float* charge' in kernel
        assert 'const float* charge_14' in kernel

        assert 'sigma_i = sigma[gi]' in kernel
        assert 'sigma_i_14 = sigma_14[gi]' in kernel
        assert 'epsilon_i = epsilon[gi]' in kernel
        assert 'charge_i = charge[gi]' in kernel

        assert 'rsqrtf' in kernel
        assert 'atomicAdd' in kernel
        assert 'energy_val' in kernel
        assert 'force_magnitude' in kernel

        open_count = kernel.count('{')
        close_count = kernel.count('}')
        assert open_count == close_count
```

- [ ] **Step 4: Run all expression tests**

Run: `conda run -n md_analysis pytest mdpy/test/test_nonbonded_expression.py -sv`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add mdpy/test/test_nonbonded_expression.py
git commit -m "test: update nonbonded expression tests for shuffle kernel structure"
```

---

### Task 6: Full Integration Validation

Run the complete test suite and OpenMM validation to confirm the new kernel produces correct results.

**Files:**
- No new files

- [ ] **Step 1: Run full test suite**

Run: `conda run -n md_analysis pytest mdpy/test/ -sv`
Expected: ALL PASS (122 tests)

- [ ] **Step 2: Run OpenMM validation (acceptance gate)**

Run: `conda run -n md_analysis pytest mdpy/test/test_openmm_validation.py -sv -k "6PO6"`
Expected: ALL PASS. These are the critical correctness checks:

| Test | What it validates |
|------|-------------------|
| `test_bonded_energy` | Bonded forces unchanged (sanity check) |
| `test_nonbonded_energy` | Nonbonded energy matches OpenMM within tolerance |
| `test_total_energy` | Total energy matches OpenMM within tolerance |
| `test_force_direction_correlation` | Force vectors point in the same direction as OpenMM |
| `test_forces_magnitude_order` | Force magnitudes are proportional to OpenMM's |

If validation fails, debug by comparing individual force components. Check these likely culprits:

1. **Exclusion mask pre-rotation**: `(excl >> tgx) | (excl << (32 - tgx))` must align with the shuffle pattern
2. **Exclusion convention**: `(excl & 0x1) != 0` means bit set = excluded (verify against `tile_list.py` mask encoding)
3. **1-4 parameter selection**: Both `sigma_i` and `sigma_j` must switch to `_14` variants when `is_14` is true
4. **Self-tile energy**: Must use `0.5f * energy_val` (each pair counted by both threads)
5. **Self-tile force**: Each thread accumulates only its own atom's force (no `shfl_fx` for self-tile)
6. **j-force writeback**: After 32 rotations, `shfl_fx` at thread tgx is the total force on `gj_init` from all i-atoms

- [ ] **Step 3: Run CharmmForcefield system test**

Run: `conda run -n md_analysis pytest mdpy/test/test_charmm_forcefield.py -sv`
Expected: ALL PASS (100-step simulation completes without NaN)

- [ ] **Step 4: Commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: integration fixes for OpenMM diagonal loop kernel"
```

---

### Task 7: Performance Benchmark

Measure ms/step with the new kernel.

**Files:**
- Create: `mdpy/test/benchmark_shuffle_kernel.py`

- [ ] **Step 1: Create benchmark script**

Create `mdpy/test/benchmark_shuffle_kernel.py`:

```python
import os
import time
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def benchmark_system(name, psf, pdb, prm, cutoff=12.0, num_steps=100):
    from mdpy.forcefield.charmm_forcefield import CharmmForcefield
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.integrator.verlet import VerletIntegrator

    ff = CharmmForcefield(psf, pdb, prm)
    system = ff.create_system(pbc_matrix=np.eye(3) * 100.0)

    integrator = VerletIntegrator(1.0)
    system.step(integrator, number_steps=3)

    start = time.perf_counter()
    system.step(integrator, number_steps=num_steps)
    elapsed = time.perf_counter() - start

    print(f'{name}: {num_steps} steps in {elapsed:.3f}s '
          f'({elapsed / num_steps * 1000:.2f} ms/step)')
    return elapsed


if __name__ == '__main__':
    psf = os.path.join(DATA_DIR, '6PO6.psf')
    pdb = os.path.join(DATA_DIR, '6PO6.pdb')
    prm = os.path.join(DATA_DIR, 'par_all36_prot.prm')

    if os.path.exists(psf):
        benchmark_system('6PO6 (49 atoms)', psf, pdb, prm, num_steps=200)
    else:
        print('Test data not found. Run from mdpy root.')
```

- [ ] **Step 2: Run benchmark**

Run: `conda run -n md_analysis python mdpy/test/benchmark_shuffle_kernel.py`
Expected: No crashes. Record ms/step.

- [ ] **Step 3: Commit**

```bash
git add mdpy/test/benchmark_shuffle_kernel.py
git commit -m "bench: add shuffle kernel benchmark script"
```

---

## Summary

| Task | What | Key Change |
|------|------|------------|
| 1 | Code generators (5 functions) | shuffle/param load/select/restore for both i and j atoms |
| 2 | Cross-tile kernel v2 | Static dispatch + diagonal loop + rotate-right-1 shuffle |
| 3 | Self-tile kernel v2 | Broadcast mode + all j (`j != tgx`) + `0.5f * energy` |
| 4 | NonbondedForce.compute() | Remove `tile_counter`, static dispatch launch config |
| 5 | Update existing tests | Fix kernel-source assertions for shuffle structure |
| 6 | Integration validation | OpenMM 6PO6 validation (acceptance gate) |
| 7 | Performance benchmark | Measure ms/step |

**atomicAdd reduction:** Cross-tile: 6 per atom (3 for gi, 3 for gj) per tile. Self-tile: 3 per atom per tile. Down from 6 per pair in the old kernel.
