# mdpy Restructure Work Plan

> Branch: `restructure`
> Env: `conda activate md_analysis`
> Validation: 6PO6 (49 atoms) for quick test, 1M9Z (95567 atoms) for benchmark

---

## Phase 1: Core Data Structures

**Goal**: 建立新架构的底层数据结构，不涉及 GPU，纯 CPU/NumPy。

**新建文件**:
| File | Description | Lines (est.) |
|------|-------------|------|
| `mdpy/core/particle_table.py` | SoA 粒子数据容器 | ~60 |
| `mdpy/core/topology.py` | Topology + Builder + CSR exclusion | ~300 |
| `mdpy/core/pbc.py` | PBC 工具函数（从 utils/pbc.py 迁移） | ~80 |
| `mdpy/force/force_term.py` | ForceTerm 基类 | ~30 |

**具体任务**:

1. **`particle_table.py`**
   - `ParticleTable.__init__(num_particles)`: 分配 positions, velocities, forces, masses, charges, particle_types, molecule_ids（全部 NumPy SoA）
   - `zero_forces()`: 清零 forces 数组
   - 使用 `env.NUMPY_FLOAT` / `env.NUMPY_INT` 精度
   - 测试：创建 100 粒子 ParticleTable，验证 shape 和 dtype

2. **`topology.py`**
   - `Topology` 类：所有数据为 compact numpy arrays，构建后不可变
     - bond/angle/dihedral/improper indices + parameters
     - CSR exclusion map（exclusion_offset, exclusion_neighbors, exclusion_scale）
     - per-atom metadata（masses, charges, particle_types）
   - `Topology.Builder` 类：
     - `set_particles(masses, charges, types, ...)`
     - `add_bond(i, j)` / `add_angle(i, j, k)` / `add_dihedral(i, j, k, l)` / `add_improper(i, j, k, l)`
     - `resolve_parameters(parameters)`: 查参数表填充 force constant 数组
     - `build_exclusion_map()`: 从 bonded topology 自动生成 CSR exclusion
       - 1-2: scale=0 (full exclude)
       - 1-3: scale=0 (full exclude)
       - 1-4: scale=1.0 (CHARMM, Coulomb no scaling)
     - `build() -> Topology`: 返回不可变对象
   - 测试：手动构造 3 原子 + 1 bond + 1 angle 拓扑，验证 exclusion map

3. **`pbc.py`**
   - 从 `utils/pbc.py` 迁移核心函数
   - `minimum_image(dx, pbc_matrix, pbc_inv)`: numba.njit
   - `wrap_position(position, pbc_matrix, pbc_inv)`: numba.njit
   - 测试：立方盒子 + 非正交盒子

4. **`force/force_term.py`**
   - `ForceTerm` 基类：`name`, `compute(gpu_context, tile_list)`, `compute_cpu(...)` (可选)

**验收标准**:
```bash
conda run -n md_analysis pytest mdpy/test/test_particle_table.py -sv
conda run -n md_analysis pytest mdpy/test/test_topology.py -sv
conda run -n md_analysis pytest mdpy/test/test_pbc.py -sv
```

**预估时间**: 3-4 天

---

## Phase 2: Bonded Force Kernels

**Goal**: 实现 bond/angle/dihedral/improper 的 GPU kernel，可独立于 Phase 3 开发。

**新建文件**:
| File | Description | Lines (est.) |
|------|-------------|------|
| `mdpy/force/bonded_force.py` | BondedForce + 4 个 CuPy RawKernel | ~400 |

**具体任务**:

1. **CUDA kernel 设计**
   - bond_kernel: `V = k(r-r0)^2`，1 thread/bond
   - angle_kernel: `V = k(θ-θ0)^2`，1 thread/angle（需向量夹角计算）
   - dihedral_kernel: `V = k(1+cos(nφ-δ))`，1 thread/dihedral（需二面角计算）
   - improper_kernel: `V = k(ψ-ψ0)^2`，1 thread/improper
   - 每个 kernel 都处理 PBC minimum image
   - 每个 kernel 用 atomicAdd 写入共享 force buffer
   - 每个 kernel 用 atomicAdd 累加 energy

2. **BondedForce 类**
   - `__init__(self, topology)`: 从 topology 取 indices + parameters，编译 4 个 kernel
   - `compute(self, gpu_context, tile_list)`: 按顺序调用 4 个 kernel
   - 跳过数量为 0 的 term（如没有 improper 就不调用 improper kernel）

3. **CPU fallback（numba.njit）**
   - 每种力一个 numba.njit 版本，用于验证 GPU 结果正确性
   - 可直接复用当前 `constraint/` 下的逻辑

**验收标准**:
```bash
conda run -n md_analysis pytest mdpy/test/test_bonded_force.py -sv
# CPU vs GPU 结果一致性
# 与现有 constraint/ 结果对比
```

**预估时间**: 4-5 天

---

## Phase 3: Tile Neighbor List

**Goal**: 实现 tile-based neighbor list 的 GPU 构建 pipeline。可与 Phase 2 并行。

**新建文件**:
| File | Description | Lines (est.) |
|------|-------------|------|
| `mdpy/core/tile_list.py` | TileList 类 + 4 个 GPU build kernel | ~500 |

**具体任务**:

1. **数据结构**
   - `tile_particles`: (num_tiles, 32) int32，不足 32 的 tile 用 -1 填充
   - `interaction_tiles_i/j`: 活跃的 tile 对列表
   - `exclusion_masks / scaling_masks`: (num_interactions, 32) uint32
   - Verlet skin: 记录 rebuild 时的 positions，检查 max displacement

2. **4 个 GPU build kernel**
   - cell_assign: 1 thread/particle，计算 cell 坐标
   - counting_sort: GPU 排序，粒子按 cell 分组
   - tile_cull: 枚举 neighbor cell pairs，检查 bounding box distance
   - build_exclusion_masks: 1 block/interaction, 32 threads/block，扫描 CSR exclusion map 生成 bitmask

3. **TileList 类**
   - `__init__(self, cutoff, skin=2.0)`
   - `rebuild(self, d_positions, topology, d_pbc_matrix)`: 调用 4 个 kernel
   - `check_rebuild(self, d_positions) -> bool`: max displacement > skin/2 时返回 True
   - 首次 rebuild 时分配所有 GPU buffer

**验收标准**:
```bash
conda run -n md_analysis pytest mdpy/test/test_tile_list.py -sv
# 验证 interaction tile 数量合理
# 验证 exclusion mask 正确标记 1-2, 1-3, 1-4
# 6PO6 (49 atoms): ~2-3 tiles, 可手动验证
```

**预估时间**: 5-6 天

---

## Phase 4: Nonbonded Expression System

**Goal**: 实现表达式系统——Python 函数 → AST 转译 → CUDA kernel。依赖 Phase 3（tile list）。

**新建文件**:
| File | Description | Lines (est.) |
|------|-------------|------|
| `mdpy/force/nonbonded_expression.py` | Parameter, @decorator, AST transpiler | ~350 |
| `mdpy/force/nonbonded_force.py` | NonbondedForce 类 | ~150 |
| `mdpy/force/expressions/__init__.py` | 导出 | ~5 |
| `mdpy/force/expressions/lennard_jones.py` | LJ 12-6 | ~15 |
| `mdpy/force/expressions/coulomb.py` | Coulomb | ~10 |
| `mdpy/forcefield/parameters.py` | ParameterTable 数据结构 | ~60 |

**具体任务**:

1. **`nonbonded_expression.py`** (核心，~350 行)
   - `Parameter` 类：支持 `__getitem__` 让 `sigma[atom_i]` 合法
   - `@nonbonded_expression` 装饰器：
     - `inspect.getsource()` 读源码
     - `ast.parse()` 解析语法树
     - 分类参数：`r`=距离, `Parameter()`=力场参数, 其余=粒子索引
     - 动态生成 kernel 签名（每个 Parameter → 一个 `const float*`）
     - 动态生成 pre-fetch 代码（每个 Parameter → `float param_i = ...; float param_j = ...;`）
     - 翻译函数体为 CUDA C 片段
     - 组装完整 kernel → `cp.RawKernel()` 编译
   - AST transpiler 规则：
     - `param[idx_var]` → `param_i` / `param_j`（pre-fetched scalar）
     - `a ** n` → 内联乘法展开
     - `sqrt(x)` → `sqrtf(x)`, `exp(x)` → `expf(x)`
     - 赋值 → `float name = value;`
     - 返回 → `*energy_out = e; *force_mag_out = f;`
   - `NonbondedExpression.__add__`：组合两个表达式
     - 变量重命名（`_2` 后缀避免冲突）
     - 合并 Parameter 声明
     - 合并 pre-fetch 代码
     - 求和 energy 和 force_magnitude

2. **`nonbonded_force.py`**
   - `NonbondedForce.__init__(expression)`: 接受 NonbondedExpression
   - `bind(topology, parameter_table, cutoff)`:
     - 按 Parameter 名字匹配 ParameterTable
     - per-type → per-atom 展开
     - 上传 GPU
     - 编译 kernel
   - `compute(gpu_context, tile_list)`: 调用 kernel

3. **`parameters.py`**
   - `ParameterTable`: `per_type: dict[str, ndarray]`, `per_atom: dict[str, ndarray]`
   - `add_per_type(name, values)`, `add_per_atom(name, values)`

4. **内置表达式**
   - `lennard_jones.py`: LJ 12-6
   - `coulomb.py`: 点电荷静电

**验收标准**:
```bash
conda run -n md_analysis pytest mdpy/test/test_nonbonded_expression.py -sv
# 测试 1: 单独 LJ 表达式转译正确
# 测试 2: 单独 Coulomb 转译正确
# 测试 3: LJ + Coulomb 组合转译正确
# 测试 4: NonbondedForce.bind() 参数匹配正确
# 测试 5: GPU kernel 执行，结果与 CPU fallback 一致
```

**预估时间**: 6-7 天

---

## Phase 5: GPUContext + System + Integrator

**Goal**: 把所有组件串联起来，实现完整的模拟循环。

**新建文件**:
| File | Description | Lines (est.) |
|------|-------------|------|
| `mdpy/core/gpu_context.py` | GPU 常驻内存管理 | ~120 |
| `mdpy/system.py` | System 类（替代 Ensemble + Simulation） | ~200 |
| `mdpy/integrator/verlet.py` | GPU Verlet integrator | ~100 |
| `mdpy/integrator/langevin.py` | GPU Langevin BAOAB | ~120 |

**具体任务**:

1. **`gpu_context.py`**
   - `GPUContext.initialize(topology, pbc_matrix)`: 分配所有 GPU buffer
     - d_positions, d_velocities, d_forces, d_prev_positions
     - d_masses, d_types
     - d_bond_indices/params, d_angle_indices/params, ...
     - d_pbc_matrix, d_pbc_inv
     - d_energy
   - `upload_positions(particle_table)`
   - `download_positions(particle_table)`
   - `download_velocities(particle_table)`

2. **`system.py`**
   - `__init__(topology, pbc_matrix, cutoff)`
   - `add_force_term(term)`: 注册 ForceTerm
   - `compute_forces()`: zero forces → 遍历 force_terms → 累积 energy
   - `step(integrator, num_steps)`: rebuild check → forces → integrate
   - `dump_state()`: GPU → CPU download

3. **Verlet integrator**
   - CuPy RawKernel: `x_new = 2*x - x_prev + a*dt^2`, PBC wrap
   - 初始化: `x_prev = x - v*dt + 0.5*a*dt^2`

4. **Langevin integrator (BAOAB)**
   - CuPy RawKernel: friction + random kicks + velocity update
   - 需要 random number generation（cuRAND 或伪随机）

**验收标准**:
```bash
conda run -n md_analysis pytest mdpy/test/test_system.py -sv
# 完整 1 步模拟: 6PO6, 验证能量非 NaN
# 完整 100 步模拟: 6PO6, 验证能量稳定
```

**预估时间**: 4-5 天

---

## Phase 6: I/O + Forcefield Adaptation

**Goal**: 让现有 Parser 和 Forcefield 产出新 API 需要的数据结构。

**修改文件**:
| File | Change |
|------|--------|
| `mdpy/io/psf_parser.py` | 输出适配 `Topology.Builder` |
| `mdpy/io/pdb_parser.py` | 输出 NumPy position 数组（而非 Quantity） |
| `mdpy/io/charmm_toppar_parser.py` | 输出 `ParameterTable` |
| `mdpy/forcefield/charmm_forcefield.py` | `create_system()` 使用新 API |
| `mdpy/dumper/*.py` | 适配 ParticleTable |
| `mdpy/analyser/*.py` | 适配 ParticleTable + Topology |

**具体任务**:

1. **PSFParser → Topology.Builder**
   - PSFParser 的 bond/angle/dihedral/improper 数据喂入 Builder
   - 单位转换在此完成（Quantity → raw float）

2. **CharmmTopparParser → ParameterTable**
   - LJ 参数 → `per_type["sigma"]`, `per_type["epsilon"]`
   - 电荷 → `per_atom["charge"]`（来自 PSF，非 PRM）

3. **CharmmForcefield.create_system()**
   - PSF + PDB + Toppar → Topology.Builder → build() → System
   - 注册 BondedForce + NonbondedForce
   - 创建 Integrator
   - 完成单位转换

4. **Dumper 适配**
   - PDBDumper / LogDumper / HDF5Dumper 使用 `system.dump_state()` 获取数据

**验收标准**:
```bash
conda run -n md_analysis pytest mdpy/test/test_charmm_forcefield.py -sv
# 完整流程: 读 PSF + PDB + PRM → create_system → 100 步模拟 → 写输出
```

**预估时间**: 4-5 天

---

## Phase 7: OpenMM Validation

**Goal**: 用 OpenMM Reference Platform 作为 ground truth 验证 mdpy 所有力的正确性。

**新建文件**:
| File | Description | Lines (est.) |
|------|-------------|------|
| `mdpy/test/generate_reference.py` | 生成 OpenMM 参考值 | ~200 |
| `mdpy/test/test_openmm_validation.py` | 对比验证 | ~300 |

**具体任务**:

1. **generate_reference.py**
   - 读 6PO6 PSF/PDB/PRM → 构建 OpenMM System
   - Reference Platform（确定性浮点）
   - NoCutoff / CutoffNonPeriodic（对齐 mdpy 无 PME）
   - 按 force group 输出：
     - bond energy, angle energy, dihedral energy, improper energy
     - nonbonded (LJ) energy, electrostatic energy
     - per-particle force vectors (N, 3)
   - 保存为 `openmm_reference_6PO6.npz`

2. **test_openmm_validation.py**
   - mdpy 计算各 force term 的 energy + force
   - 与参考值对比，检查容忍度：
     | Force term | Energy relative error | Force absolute error |
     |------------|----------------------|---------------------|
     | bond | < 1e-6 | < 1e-4 |
     | angle | < 1e-6 | < 1e-4 |
     | dihedral | < 1e-6 | < 1e-4 |
     | improper | < 1e-6 | < 1e-4 |
     | nonbonded (LJ) | < 1e-3 | < 1e-2 |
     | electrostatic | < 1e-3 | < 1e-2 |

**验收标准**:
```bash
conda run -n md_analysis python mdpy/test/generate_reference.py
conda run -n md_analysis pytest mdpy/test/test_openmm_validation.py -sv -k "6PO6"
```

**预估时间**: 3-4 天

---

## Phase 8: Optimization + Polish

**Goal**: 性能优化 + 清理旧代码。

**具体任务**:

1. **int64 Q32.32 fixed-point force accumulation**: 减少 float32 atomicAdd 精度损失
2. **Morton sort**: 按空间位置排序粒子，提高 tile 内空间局部性
3. **Warp-level reduce**: 优化 force accumulation 的 GPU 吞吐
4. **`__restrict__`**: 所有 kernel pointer params 加 restrict
5. **Dumper/Analyzer 适配**: 完善所有输出和分析工具
6. **Example scripts**: 更新 `example/charmm/charmm.py`
7. **删除旧代码**:
   - `core/particle.py`, `core/cell_list.py`
   - `constraint/` (entire directory)
   - `ensemble.py`, `simulation.py`
   - `utils/geometry.py`, `utils/pbc.py`

**验收标准**:
```bash
# 1M9Z (95567 atoms) 完整模拟
conda run -n md_analysis pytest mdpy/test/test_openmm_validation.py -sv -m "slow"
# 性能基准测试: 对比 restructure 前后的每步耗时
```

**预估时间**: 5-7 天

---

## Phase Dependencies

```
Phase 1: Core Data Structures
  │
  ├──(parallel)──────────────────┐
  ▼                              ▼
Phase 2:                       Phase 3:
Bonded Force Kernels           Tile Neighbor List
  │                              │
  │                              ▼
  │                            Phase 4:
  │                            Nonbonded Expression System
  │                              │
  ▼                              ▼
Phase 5: GPUContext + System + Integrator
  │
  ▼
Phase 6: I/O + Forcefield Adaptation
  │
  ▼
Phase 7: OpenMM Validation
  │
  ▼
Phase 8: Optimization + Polish
```

**Phase 2 和 Phase 3 可以并行开发**，它们只依赖 Phase 1，互相不依赖。

---

## Time Estimate

| Phase | Days | Cumulative |
|-------|------|-----------|
| 1. Core Data Structures | 3-4 | 3-4 |
| 2. Bonded Force Kernels | 4-5 | 7-9 |
| 3. Tile Neighbor List (parallel w/ 2) | 5-6 | 7-9 |
| 4. Nonbonded Expression System | 6-7 | 13-16 |
| 5. GPUContext + System + Integrator | 4-5 | 17-21 |
| 6. I/O + Forcefield Adaptation | 4-5 | 21-26 |
| 7. OpenMM Validation | 3-4 | 24-30 |
| 8. Optimization + Polish | 5-7 | 29-37 |

**Total: ~30-37 working days** (parallelizing Phase 2+3 saves ~5 days)
