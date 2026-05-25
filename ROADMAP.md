# mdpy GPU Nonbonded Force Roadmap

## Current Status (as of 2026-05)

mdpy 已经实现了基于 OpenMM 32-atom tile 模式的 nonbonded GPU 计算：

- `TileList`: 32-atom block 分组，bounding box 粗筛 + O(N^2) CPU 端 tile pair 构建
- `NonbondedExpression`: Python 函数 → AST 转译 → CUDA C fragment → `cp.RawKernel` JIT 编译
- Kernel: 256 threads/block，每个 thread 串行处理 4 个 pair (128 pairs / thread)，6 次 `atomicAdd` / pair
- Exclusion: bitmask 编码 (32-bit per row)，exclusion + 1-4 scaling 分离存储
- 支持 PBC minimum image

---

## GROMACS vs OpenMM 技术路线分析

### 架构对比

| 维度 | GROMACS nbnxm (8x8 cluster) | OpenMM (32x32 tile) | mdpy 当前实现 |
|------|---------------------------|---------------------|--------------|
| 原子分组 | 8 atoms/cluster, 8 clusters/super-cluster | 32 atoms/block (匹配 warp size) | 32 atoms/tile (同 OpenMM) |
| 基本计算单元 | 1 cluster pair = 64 pairs | 1 tile = 1024 pairs | 1 tile (当前串行 128 pairs/thread) |
| 线程映射 | 1 thread = 1 (i,j) pair | 1 thread = 1 i-atom vs 32 j-atoms (对角线循环) | 1 thread = 4 pairs (线性遍历) |
| Thread block | 64 threads (8x8) | 256 threads (8 warps) | 256 threads (当前) |
| i-atom 数据 | shared memory 预加载 | register 持有 (1 个 i-atom) | shared memory 预加载 |
| j-atom 数据 | global memory 流式加载 | warp shuffle 旋转传递 | shared memory 预加载 |
| Force 归约 | warp shuffle `__shfl_down` 求和 | 对角线循环 + shuffle 旋转 | 朴素 `atomicAdd` per pair |
| Exclusion | bitmask + j-entry 排序优化 | bitmask + 独立 exclusion tile 列表 | bitmask per row |
| Electrostatics | Cut / RF / Ewald解析 / Ewald查表 / TwinCut | NoCutoff / RF / PME / LJPME | Plain Coulomb (direct only) |
| VdW | Cut / CombGeom / CombLB / ForceSwitch / PotSwitch / EwaldGeom / EwaldLB | Cut / PotentialSwitch / LJPME | 由 NonbondedExpression 定义 |
| 动态剪枝 | inner/outer cutoff + GPU prune kernel | padded cutoff + 延迟重建 | skin distance + rebuild 检测 |

### 性能预期 (单 GPU)

| 体系规模 | OpenMM tile | GROMACS cluster | 差异 |
|---------|------------|----------------|------|
| < 1 万原子 | 优秀 | 良好 | < 5% |
| 1-10 万原子 | 优秀 | 优秀 | < 5% |
| 10-50 万原子 | 优秀 | 优秀 | ~5% |
| > 50 万原子 | 良好 | 优秀 (sorting + pruning) | 5-15% |

**结论**: 单 GPU 场景下两者性能持平，选择主要取决于实现复杂度和体系规模偏好。

### 选型结论

mdpy 选择 OpenMM 32-atom tile 模式，理由：

1. **已有实现基础**: `TileList` + `NonbondedExpression` 已基于此模式
2. **负载均衡**: tile 粒度均匀 (每 tile 固定 1024 pairs)，GPU SM 间负载一致
3. **实现简洁**: "每个 thread 持有 1 个 i-atom，对角线循环 vs 32 个 j-atom" 的模式清晰
4. **小体系友好**: mdpy 的目标体系 (几十到几万原子) 更适合 32-atom 分组
5. **CuPy RawKernel 兼容**: 对角线循环 + `__shfl_sync` 在 RawKernel 中完全可用

---

## 术语严格区分

| 术语 | 所属领域 | 含义 |
|------|---------|------|
| **Atom Block** | MD 层面 | 32 个原子的分组，原子空间划分的基本单位 |
| **Tile** | MD 层面 | **两个** Atom Block 之间的交互对，即 (block_x, block_y)，代表 32×32 = 1024 个原子对 |
| **Thread Block** (GPU Block) | GPU 层面 | CUDA 编程中的 `blockDim`，256 个 thread，8 个 warp，是 SM 调度的基本单位 |
| **Warp** | GPU 层面 | 32 个 thread，锁步执行，是 SM 执行的最小单位 |

**Tile ≠ GPU Block**。Tile 是 MD 的逻辑任务（1024 个原子对的计算），GPU Block 是硬件调度单位（256 个 thread）。

---

## OpenMM Kernel 执行模型详解

### Host 端：确定任务总量

```
100,000 个原子
→ numAtomBlocks = ceil(100000 / 32) = 3125 个 Atom Block
→ 全量 Tile 数 = 3125 × 3126 / 2 = 4,882,875 个 Tile (上三角 + 对角线)

如果使用 cutoff + 邻居列表:
→ 邻居列表 kernel 筛选出距离 < cutoff 的 Tile
→ 假设筛选后 interactionCount = 50,000 个 Tile 需要计算

这 50,000 个 Tile 就是任务总量。
```

### Host 端：决定 GPU Grid 配置

CUDA kernel 启动语法：

```cuda
computeNonbonded<<<blocks_per_grid, threads_per_block>>>(参数);
```

这两个参数**完全由 Host 端代码在启动 kernel 之前决定**，GPU 硬件不会自动选择。

OpenMM 的决定方式：

```cpp
// CudaNonbondedUtilities.cpp 构造函数，初始化时就定好了
int multiprocessors;
cuDeviceGetAttribute(&multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
numForceThreadBlocks = 4 * multiprocessors;   // blocks_per_grid
forceThreadBlockSize = 256;                    // threads_per_block

// 实际启动：
context.executeKernel(kernel, args,
    numForceThreadBlocks * forceThreadBlockSize,  // total_threads
    forceThreadBlockSize);                        // threads_per_block
```

**Grid 配置只由 GPU 硬件（SM 数量）决定，不随体系大小变化。** 这是一个设计选择。

```
GPU 有 80 个 SM → blocks_per_grid = 4 × 80 = 320
每个 GPU Block = 256 threads = 8 个 warp
Grid 总 warp 数 = 320 × 8 = 2,560
```

### Kernel 内部：每个 Warp 自行认领任务

每个 warp 通过全局 warp 编号和一个整数除法自行算出自己负责哪段 Tile：

```cuda
const unsigned int totalWarps = (blockDim.x * gridDim.x) / 32;  // = 2560
const unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
const unsigned int tgx = threadIdx.x & 31;
const unsigned int tbx = threadIdx.x - tgx;

// 从 50,000 个 Tile 中领取自己的一段:
int pos = warp * 50000 / 2560;
int end = (warp+1) * 50000 / 2560;

// Warp 0:    pos=0,     end=19     → 19 个 Tile
// Warp 1:    pos=19,    end=39     → 20 个 Tile
// ...
// Warp 2559: pos=49980, end=50000  → 20 个 Tile
```

**一次分发，不是动态分发。** 没有 Host 端循环派发，没有任务队列，没有动态调度。

### 为什么用静态一次分发

**动态分发**（有个任务队列，warp 做完一个来领下一个）：

```
需要一个全局计数器: "下一个待处理的 Tile 编号"
每个 warp 做完后: atomicAdd(&nextTask, 1) → 取新任务
问题:
  - atomicAdd 是串行操作, 所有 warp 抢同一个计数器 → 瓶颈
  - 需要全局同步或锁 → GPU 不擅长
  - 额外的 global memory 读写 → 浪费带宽
  - 实现复杂
```

**静态一次分发**（每个 warp 用除法自己算任务范围）：

```
每个 warp 只需一次整数除法: pos = warp * totalTiles / totalWarps
好处:
  1. 零开销: 一个除法 vs 每次任务完成后的 atomicAdd
  2. 无竞争: 每个 warp 独立工作, 不需要访问任何共享计数器
  3. 内存友好: Tile 列表是只读的, 所有 warp 顺序读取, GPU cache 命中率高
  4. 简单: 几行代码搞定
```

缺点：负载不完全均衡（有的 warp 分到 20 个 Tile，有的 19 个），但差异最多 1 个，可忽略。

### Warp 处理多个 Tile 的时序

```
Warp 0 被分配到某个 SM 上开始执行:

  pos=0, end=19 → 需要处理 19 个 Tile

  [取 Tile[0]: x=5, y=3]
  [global load: posq[160..191] + posq[96..127]]
  [32步循环: 计算 1024 pairs]
  [atomicAdd 写回]
  [取 Tile[1]: x=12, y=7]
  [global load]
  [32步循环]
  [atomicAdd 写回]
  ...
  [取 Tile[18]: x=45, y=30]
  [32步循环]
  [atomicAdd 写回]
  [pos == end → 结束]
```

2560 个 warp 各自串行处理自己的 Tile，全部完成后 kernel 返回。

### Warp 内代码逐行详解

以下用一个具体 Tile 为例：`(x=5, y=3)`，即 block_5（原子 160..191）和 block_3（原子 96..127）的交互。

```cuda
int x = tiles[pos].x;   // = 5
int y = tiles[pos].y;   // = 3
```

`tiles[pos]` 是一个 `int2`，存了两个 Atom Block 的编号。32 个 thread 读到相同的 x 和 y。

```cuda
atom1 = x * 32 + tgx;
```

"我的 i-atom 是谁"。`x * 32` 是 block_x 的起始原子编号，`tgx` 是我在 warp 中的编号 (0..31)。

```
Thread 0  (tgx=0):  atom1 = 5×32 + 0  = 160  ← block_5 的第 0 个原子
Thread 1  (tgx=1):  atom1 = 5×32 + 1  = 161
...
Thread 31 (tgx=31): atom1 = 5×32 + 31 = 191  ← block_5 的第 31 个原子

32 个 thread 各自认领 block_5 中的一个原子, 每个 thread 是该原子的"代理人"。
```

```cuda
posq1 = posq[atom1];
```

从 global memory 读取我负责的 i-atom 的坐标。32 个 thread 并行读 32 个连续位置，合并读取（coalesced），一次内存事务。`posq1` 存在 register 中，整个 Tile 生命周期内不变。

```cuda
shflPosq = posq[y * 32 + tgx];
```

初始加载 j-atom 坐标：

```
Thread 0  (tgx=0):  shflPosq = posq[3×32 + 0]  = posq[96]   ← block_3[0]
Thread 1  (tgx=1):  shflPosq = posq[3×32 + 1]  = posq[97]   ← block_3[1]
...
Thread 31 (tgx=31): shflPosq = posq[3×32 + 31] = posq[127]  ← block_3[31]
```

初始时 thread_i 持有 block_y 的第 i 个原子。后面会通过 SHUFFLE 在 warp 中传递。

```cuda
shflForce = {0, 0, 0};
force = {0, 0, 0};
```

两个力累加器清零。`force` 留在 register（i-atom 受力），`shflForce` 跟着 j-atom 数据一起 shuffle。

```cuda
unsigned int tj = tgx;
```

`tj` 记录"我当前持有的 j-atom 是 block_y 中第几个"。初始 `tj = tgx`，后面每步 +1。

#### 32 步循环

```cuda
for (int j = 0; j < 32; j++) {
    delta = shflPosq - posq1;
    r = |delta|;
    dEdR = force_function(r);
```

每步计算一个 pair 的距离和力大小。32 个 thread × 32 步 = 1024 个 pair。

```cuda
    force    -= delta * dEdR;
    shflForce += delta * dEdR;
```

Newton 第三定律：`force` 累积 i-atom 受力（留在 register），`shflForce` 累积 j-atom 受力（跟着 shuffle）。两者大小相等、方向相反。

```cuda
    SHUFFLE_WARP_DATA;
```

展开后实际是：

```cuda
shflPosq.x = __shfl_sync(shflPosq.x, tgx+1);
shflPosq.y = __shfl_sync(shflPosq.y, tgx+1);
shflPosq.z = __shfl_sync(shflPosq.z, tgx+1);
shflForce.x = __shfl_sync(shflForce.x, tgx+1);
shflForce.y = __shfl_sync(shflForce.y, tgx+1);
shflForce.z = __shfl_sync(shflForce.z, tgx+1);
```

`__shfl_sync(var, srcLane)` 语义："我从 warp 内第 srcLane 个 thread 那里取 var 的值"。

`srcLane = tgx+1` 的效果：rotate-right-1

```
Thread 0  (tgx=0):  从 thread 1  取值 → 拿到 thread 1  的 j-atom 坐标和力
Thread 1  (tgx=1):  从 thread 2  取值 → 拿到 thread 2  的数据
...
Thread 30 (tgx=30): 从 thread 31 取值 → 拿到 thread 31 的数据
Thread 31 (tgx=31): 从 thread 0  取值 → wrap around (31+1=32 mod 32=0)
```

每一步，每个 thread 手中的 j-atom 数据（坐标 + 已累积的力）向右移动一位。32 步后绕一圈回到原位。

```cuda
    tj = (tj + 1) & 31;
```

`& 31` 等价于 `% 32`（因为 32 是 2 的幂）。tj 跟 shuffle 同步：shuffle 把 j-atom_1 的数据送到 thread 0 的同时，tj 也变成 1，所以 thread 知道"我现在拿着的是 block_y 的第几个原子"。

#### Thread 0 (tgx=0) 的完整视角

```
我的身份: tgx=0, i-atom = 原子 160 (block_5[0])
我负责: 累积原子 160 受到的所有来自 block_3 的力

Step 0: 持有 j-atom_96, 计算 (160, 96)
        force    -= F(160←96)     ← 原子160的力, 留在我这里
        shflForce += F(96←160)    ← 原子96的力, 跟着数据走
        Shuffle: 原子96的数据 →送给 thread 31, 我收到 thread 1 的原子97的数据

Step 1: 持有 j-atom_97, 计算 (160, 97)
        force    -= F(160←97)
        shflForce += F(97←160)
        Shuffle: 原子97的数据 → 送给 thread 31, 我收到原子98的数据

...

Step 31: 持有 j-atom_127, 计算 (160, 127)
         force    -= F(160←127)   ← 原子160的总力完整了!
         shflForce += F(127←160)
         Shuffle: 原子96的 shflForce 经历32步回到 thread 0!
                  = F(96←160)+F(96←161)+...+F(96←191) = 原子96的总力!
```

#### 写回

```cuda
atomicAdd(&forceBuffers[x*32 + tgx], force);       // i-force
atomicAdd(&forceBuffers[y*32 + tgx], shflForce);   // j-force
```

- `x*32 + tgx` = block_x 的第 tgx 个原子的全局编号 → 写 i-force
- `y*32 + tgx` = block_y 的第 tgx 个原子的全局编号 → 写 j-force
- 写回地址用 `tgx`（thread 的永久 lane index），不是 `tj`（循环变量），因为 32 步 shuffle 后 shflForce 恰好回到最初持有它的 thread

```
Thread 0:  forceBuffers[160] += force       ← 原子160 受到的来自 block_3 的总力
           forceBuffers[96]  += shflForce   ← 原子96  受到的来自 block_5 的总力
Thread 1:  forceBuffers[161] += force       ← 原子161
           forceBuffers[97]  += shflForce   ← 原子97
...
```

### Grid 配置 vs Tile 数量

```
Grid 配置 = 4 × SM数量 (固定, 由 GPU 硬件决定)
Tile 数量 = 邻居列表筛选后的交互 Tile 数 (变化, 由体系和 cutoff 决定)

Tile >> Warp 数: 每个 warp 处理多个 Tile, GPU 充分利用 (大体系, 正常情况)
Tile ≈ Warp 数: 每个 warp 处理 ~1 个 Tile (中等体系)
Tile < Warp 数: 部分 warp 空转 (极小体系, 浪费但结果正确)
```

---

## Force Reduction 策略详解

Nonbonded kernel 的性能瓶颈不在计算（FLOPs），而在 force 写回（atomicAdd）。
三种方案的核心区别在于：**何时、如何将多个 thread 产生的力合并为一个原子的最终受力**。

---

### 方案 A: 朴素 atomicAdd (当前 mdpy 实现)

#### 数据布局

```
Tile = (block_i, block_j), 每个 block 有 32 个原子
Thread block: 256 threads
每个 thread 负责 4 个 pair (线性遍历 32×32 = 1024 pairs / 256 threads)
```

#### 计算流程 (每个 pair)

```
Step 1: 计算 pair (atom_gi, atom_gj) 的力
        fx = force_magnitude * dx / r
        fy = force_magnitude * dy / r
        fz = force_magnitude * dz / r

Step 2: 直接 atomicAdd 写回 global memory
        atomicAdd(&forces[gi*3+0],  fx);   // i-atom x
        atomicAdd(&forces[gi*3+1],  fy);   // i-atom y
        atomicAdd(&forces[gi*3+2],  fz);   // i-atom z
        atomicAdd(&forces[gj*3+0], -fx);   // j-atom x (Newton 3rd)
        atomicAdd(&forces[gj*3+1], -fy);   // j-atom y
        atomicAdd(&forces[gj*3+2], -fz);   // j-atom z
```

#### 问题

```
每个 pair: 6 次 atomicAdd
每个 tile: 1024 pairs × 6 = 6144 次 atomicAdd

同一 atom 可能被多个 thread 同时写入:
  atom_gi 被 ~32 个 thread atomicAdd (该 atom 与 block_j 中 32 个 j-atom 都交互)
  atom_gj 也被 ~256 个 thread atomicAdd (该 atom 在 block_j 中, 被 block_i 的 32 个 i-atom 交互)

atomicAdd 串行化 → GPU 必须排队等前一个写完才能写下一个
```

---

### 方案 B: GROMACS `__shfl_down` 求和归约

#### 数据布局

```
Super-cluster: 8 clusters × 8 atoms = 64 atoms
Thread block: 8×8 = 64 threads (dim3(8, 8, 1))
每个 thread 由 (tidxi, tidxj) 标识:
  tidxi = threadIdx.x  (0..7): 对应 i-cluster 内第 tidxi 个 atom
  tidxj = threadIdx.y  (0..7): 对应 j-cluster 内第 tidxj 个 atom
  → thread(tidxi, tidxj) 负责 (i_atom[tidxi], j_atom[tidxj]) 这一对
```

#### 力的产生：一个 j-atom 的力分散在 8 个 thread 中

```
处理 cluster pair (ci, cj) 时:
  j-atom = cj × 8 + tidxj    ← 8 个 thread 共享同一个 j-atom index (tidxj 相同)

  循环 8 个 i-cluster:
    thread(0, tidxj): 计算 f(j ← i_cluster_0[tidxi=0])  → fCjBuf 累加
    thread(1, tidxj): 计算 f(j ← i_cluster_1[tidxi=1])  → fCjBuf 累加
    ...
    thread(7, tidxj): 计算 f(j ← i_cluster_7[tidxi=7])  → fCjBuf 累加

  循环结束后, fCjBuf 持有该 thread 对 j-atom 的力贡献。
  但同一个 j-atom 的总力分散在 tidxi=0..7 共 8 个 thread 的 fCjBuf 中!
```

用 4-atom cluster 简化示例（实际是 8-atom）：

```
处理 j-atom_0 (tidxj=0 的所有 thread):

  thread(0,0): fCjBuf = f(j0 ← i0) + f(j0 ← i0') + ...  (遍历 8 个 i-cluster)
  thread(1,0): fCjBuf = f(j0 ← i1) + f(j0 ← i1') + ...
  thread(2,0): fCjBuf = f(j0 ← i2) + f(j0 ← i2') + ...
  thread(3,0): fCjBuf = f(j0 ← i3) + f(j0 ← i3') + ...

  需要: F_j0_total = thread(0,0).fCjBuf + thread(1,0).fCjBuf
                      + thread(2,0).fCjBuf + thread(3,0).fCjBuf
```

#### Shuffle 归约：把 8 个值合并为 1 个

GROMACS 使用 `__shfl_down` 实现 tree reduction。
以 8 个 thread (lane 0..7) 归约 fx 为例：

```
初始状态 (每个 thread 的 register):
  lane 0: fx = f0
  lane 1: fx = f1
  lane 2: fx = f2
  lane 3: fx = f3
  lane 4: fx = f4
  lane 5: fx = f5
  lane 6: fx = f6
  lane 7: fx = f7

__shfl_down(fx, 1) 的含义:
  每个 lane 获得 lane+1 的值并加到自己身上
  (最高 lane 获得无效值 0)

Round 1: offset = 1
  lane 0: fx += shfl_down(fx, 1) → fx = f0 + f1
  lane 1: fx += shfl_down(fx, 1) → fx = f1 + f2
  lane 2: fx += shfl_down(fx, 1) → fx = f2 + f3
  lane 3: fx += shfl_down(fx, 1) → fx = f3 + f4
  lane 4: fx += shfl_down(fx, 1) → fx = f4 + f5
  lane 5: fx += shfl_down(fx, 1) → fx = f5 + f6
  lane 6: fx += shfl_down(fx, 1) → fx = f6 + f7
  lane 7: fx += 0                 → fx = f7 (忽略)

Round 2: offset = 2
  lane 0: fx += shfl_down(fx, 2) → fx = (f0+f1) + (f2+f3) = f0+f1+f2+f3
  lane 1: fx += shfl_down(fx, 2) → fx = (f1+f2) + (f3+f4) = f1+f2+f3+f4
  lane 2: fx += shfl_down(fx, 2) → fx = (f2+f3) + (f4+f5) = f2+f3+f4+f5
  lane 3: fx += shfl_down(fx, 2) → fx = (f3+f4) + (f5+f6) = f3+f4+f5+f6
  lane 4-7: 部分和 (不需要)

Round 3: offset = 4
  lane 0: fx += shfl_down(fx, 4) → fx = (f0+f1+f2+f3) + (f4+f5+f6+f7)
                                      = f0+f1+f2+f3+f4+f5+f6+f7  ← 完整总和!
  lane 1-7: 部分和 (不需要)
```

3 轮后 `lane 0` 持有完整 j-force。实际代码（`nbnxm_hip_kernel_body.h:332-355`）：

```cuda
// GROMACS j-force reduction (简化版, 实际用 AMD DPP 指令优化)
template<PairlistType pairlistType>
__device__ float reduceForceJWarpShuffle(AmdPackedFloat3 f, int tidxi) {
    // f[0]=fx, f[1]=fy, f[2]=fz 是同一 j-atom 的 3 个力分量
    f[0] += __shfl_down(f[0], 1);  // Round 1
    f[1] += __shfl_down(f[1], 1);
    f[2] += __shfl_down(f[2], 1);

    f[0] += __shfl_down(f[0], 2);  // Round 2
    f[1] += __shfl_down(f[1], 2);
    f[2] += __shfl_down(f[2], 2);

    f[0] += __shfl_down(f[0], 4);  // Round 3
    // ...

    return f[0];  // 只有 tidxi==0 的 thread 持有完整值
}
```

#### 写回

```cuda
// 只有 tidxi ∈ {0,1,2} 的 3 个 thread 各写一个分量
const float reducedForceJ = reduceForceJWarpShuffle(fCjBuf, tidxi);
if (tidxi < 3) {
    amdFastAtomicAddForce(gm_f, aj, tidxi, reducedForceJ);
    // aj 是 j-atom 全局 index, tidxi 决定 x/y/z 分量
    // 只有 3 次 atomicAdd (而非 24 次)
}
```

#### i-force 处理

```
i-force 用 fCiBuffer[8] 在 register 中跨所有 j-cluster 累积:
  循环所有 jPacked:
    循环所有 j-cluster:
      循环 8 个 i-cluster:
        fCiBuffer[i] -= forceIJ;   // 每步只在 register 中加减

全部循环结束后才做一次归约 + atomicAdd:
  reduceForceI() 对 fCiBuffer[8] 做 shuffle 归约
  每个 i-atom 只写 3 次 atomicAdd (x/y/z)
```

#### GROMACS 方案汇总

```
cluster pair = 8×8 = 64 pairs:
  j-force: 3 轮 shuffle → 3 次 atomicAdd / j-atom
  i-force: register 累积 + 最终 3 轮 shuffle → 3 次 atomicAdd / i-atom
  总 atomicAdd: (8 i-atoms + 8 j-atoms) × 3 = 48 次 / 64 pairs
  → 0.75 次 atomicAdd / pair
```

---

### 方案 C: OpenMM 对角线循环 + shuffle 旋转 (推荐)

#### 数据布局

```
Tile = (block_x, block_y), 每个 block 有 32 个原子
1 个 warp (32 threads) 处理 1 个 tile
tgx = threadIdx.x & 31  (lane id, 0..31)
每个 thread 固定持有 1 个 i-atom: atom1 = block_x × 32 + tgx
```

#### 核心思想：j-atom 数据在 warp 中旋转

OpenMM 不需要显式 reduction。它让 j-atom 的坐标和力在 warp 中**循环传递**，
每个 thread 在 32 步中依次"见到"所有 32 个 j-atom。

用 4-atom tile 简化示例（实际是 32-atom）：

```
Warp 有 4 个 thread (实际 32 个):
  T0 持有 atom_a (i-atom, 固定不动)
  T1 持有 atom_b
  T2 持有 atom_c
  T3 持有 atom_d

block_x = [a, b, c, d]
block_y = [e, f, g, h]

Step 0 — 每个 thread 初始持有同 index 的 j-atom:
  T0 持有 j_data[e], T1 持有 j_data[f], T2 持有 j_data[g], T3 持有 j_data[h]

  T0 计算 (a, e) → force  += F(a,e)  (自己的 i-atom 受力, 累积在 register)
                  shflForce += F(a,e)  (当前 j-atom 受力, 累积在 register)
                  注意: force 和 shflForce 方向相反 (Newton 3rd)

  T1 计算 (b, f)
  T2 计算 (c, g)
  T3 计算 (d, h)

  SHUFFLE: 所有 j_data 向右旋转 1 位:
    T0 ← T1 的 j_data[f]
    T1 ← T2 的 j_data[g]
    T2 ← T3 的 j_data[h]
    T3 ← T0 的 j_data[e]  (wrap around)

Step 1:
  T0 持有 j_data[f] → 计算 (a, f)
  T1 持有 j_data[g] → 计算 (b, g)
  T2 持有 j_data[h] → 计算 (c, h)
  T3 持有 j_data[e] → 计算 (d, e)

  SHUFFLE: 再旋转 1 位

Step 2:
  T0 持有 j_data[g] → 计算 (a, g)
  T1 持有 j_data[h] → 计算 (b, h)
  T2 持有 j_data[e] → 计算 (c, e)
  T3 持有 j_data[f] → 计算 (d, f)

Step 3:
  T0 持有 j_data[h] → 计算 (a, h)
  T1 持有 j_data[e] → 计算 (b, e)
  T2 持有 j_data[f] → 计算 (c, f)
  T3 持有 j_data[g] → 计算 (d, g)

4 步结束! 每个 thread 计算了 4 个 pair:
  T0: (a,e), (a,f), (a,g), (a,h) — i-atom a 的所有交互
  T1: (b,f), (b,g), (b,h), (b,e) — i-atom b 的所有交互
  T2: (c,g), (c,h), (c,e), (c,f) — i-atom c 的所有交互
  T3: (d,h), (d,e), (d,f), (d,g) — i-atom d 的所有交互
```

#### SHUFFLE_WARP_DATA 的精确语义

`SHUFFLE_WARP_DATA` 由 `CudaNonbondedUtilities.cpp` 中的 `createInteractionKernel()` 动态生成：

```cuda
// 源码: CudaNonbondedUtilities.cpp, createInteractionKernel() 中
shuffleWarpData << "shflPosq.x = real_shfl(shflPosq.x, tgx+1);\n";
shuffleWarpData << "shflPosq.y = real_shfl(shflPosq.y, tgx+1);\n";
shuffleWarpData << "shflPosq.z = real_shfl(shflPosq.z, tgx+1);\n";
shuffleWarpData << "shflPosq.w = real_shfl(shflPosq.w, tgx+1);\n";
shuffleWarpData << "shflForce.x = real_shfl(shflForce.x, tgx+1);\n";
shuffleWarpData << "shflForce.y = real_shfl(shflForce.y, tgx+1);\n";
shuffleWarpData << "shflForce.z = real_shfl(shflForce.z, tgx+1);\n";
// + 所有 force parameters 也一起旋转 (如 shflSigmaEpsilon)
```

其中 `real_shfl(var, srcLane)` 封装了 `__shfl_sync(0xffffffff, var, srcLane)`。

**语义**: `thread i` 获取 `lane (tgx+1)` 处 `var` 的值。由于 `tgx` 是 `thread i` 的 lane index，
`tgx+1` 就是 `lane(i+1)`。当 `i=31` 时，`srcLane=32` 对 `__shfl_sync` 取模后等于 `lane 0`。

**效果**: rotate-right-by-1（每个 thread 从右边邻居获取数据，最右端 wrap 到最左端）：

```
T0 ← T1 的值
T1 ← T2 的值
T2 ← T3 的值
...
T30 ← T31 的值
T31 ← T0 的值     (wrap around)
```

**关键**: `shflPosq`（坐标）和 `shflForce`（力）一起旋转。j-atom 的坐标、参数、已累积的力
作为一组数据在 warp 中循环传递。

**与 BROADCAST_WARP_DATA 的对比**:

```
SHUFFLE_WARP_DATA (非对角线 tile, 旋转模式):
  shflPosq.x = real_shfl(shflPosq.x, tgx+1);   // 每个 thread 从不同的 source lane 获取
  // T0 从 lane1, T1 从 lane2, ..., T31 从 lane0 → rotate-right-1

BROADCAST_WARP_DATA (对角线 tile, 广播模式):
  posq2.x = real_shfl(shflPosq.x, j);           // 所有 thread 从同一个 source lane 获取
  // j 是循环变量, 所有 32 个 thread 都看到 lane j 的数据 → 广播
```

---

#### 对角线 tile (x == y): 广播模式，无 shflForce

对角线 tile 中 block_x == block_y，pair (i,j) 会被计算两次（thread i 算一次，
thread j 算一次）。OpenMM 用 `interactionScale = 0.5` 处理重复，只用广播，不用旋转。

```cuda
if (x == y) {
    real4 shflPosq = posq1;             // 初始: 每个 thread 持有自己的 i-atom 坐标
    // 注意: 没有 shflForce!

    for (unsigned int j = 0; j < TILE_SIZE; j++) {
        BROADCAST_WARP_DATA              // 所有 thread 从 lane j 广播 → posq2
        // 计算 (atom_i, atom_j) 的交互
        const real interactionScale = 0.5f;
        COMPUTE_INTERACTION
        force.x -= delta.x * dEdR;      // 只累积 i-force
        // 没有 shflForce 累加!
    }
    // 循环结束后只写 i-force, 不写 j-force
}
```

**为什么不需要 shflForce**: pair (a, b) 被计算两次——thread_a 在某步广播看到 b 时算一次，
thread_b 在某步广播看到 a 时算一次。两次都乘以 0.5。每个 thread 只累积自己的 `force`
（i-force），写回时 `force` 已是正确值。

---

#### 非对角线 tile (x != y): 旋转模式 — 完整推导

以下用 TILE_SIZE=4 的简化示例严格推导 shflForce 归约过程。

**设定**:
- block_x = [a, b, c, d], block_y = [e, f, g, h]
- 4 个 thread: T0(tgx=0), T1(tgx=1), T2(tgx=2), T3(tgx=3)
- 每个 thread 固定持有 1 个 i-atom: T0→a, T1→b, T2→c, T3→d
- Newton 第三定律: `force -= delta*dEdR`（i-atom 受力）, `shflForce += delta*dEdR`（j-atom 受力）

**循环前的初始状态**:

```
Thread | i-atom | shflPosq (j-atom) | shflForce    | tj
-------|--------|--------------------|--------------|----
T0     | a      | posq[e]            | (0, 0, 0)    | 0
T1     | b      | posq[f]            | (0, 0, 0)    | 1
T2     | c      | posq[g]            | (0, 0, 0)    | 2
T3     | d      | posq[h]            | (0, 0, 0)    | 3
```

##### Iteration j=0

**计算**（SHUFFLE 之前）:

```
T0: pair (a, e) → force_a    += F(a←e)
                   shflForce_e += F(e←a)       ← 记为 S_e = F(e←a)

T1: pair (b, f) → force_b    += F(b←f)
                   shflForce_f += F(f←b)       ← 记为 S_f = F(f←b)

T2: pair (c, g) → force_c    += F(c←g)
                   shflForce_g += F(g←c)       ← 记为 S_g = F(g←c)

T3: pair (d, h) → force_d    += F(d←h)
                   shflForce_h += F(h←d)       ← 记为 S_h = F(h←d)
```

**SHUFFLE_WARP_DATA**（rotate-right-1: T_i ← T_{i+1 mod 4}）:

```
T0 ← T1: shflPosq = posq[f], shflForce = S_f = F(f←b)
T1 ← T2: shflPosq = posq[g], shflForce = S_g = F(g←c)
T2 ← T3: shflPosq = posq[h], shflForce = S_h = F(h←d)
T3 ← T0: shflPosq = posq[e], shflForce = S_e = F(e←a)

tj 更新: T0: tj=1, T1: tj=2, T2: tj=3, T3: tj=0
```

**Iteration j=0 结束后的状态**:

```
Thread | shflPosq | shflForce        | tj | 该 shflForce 属于哪个 j-atom
-------|----------|------------------|----|-----------------------------
T0     | posq[f]  | F(f←b)          | 1  | f (来自 T1 的初始数据)
T1     | posq[g]  | F(g←c)          | 2  | g (来自 T2 的初始数据)
T2     | posq[h]  | F(h←d)          | 3  | h (来自 T3 的初始数据)
T3     | posq[e]  | F(e←a)          | 0  | e (来自 T0 的初始数据)
```

##### Iteration j=1

**计算**:

```
T0: pair (a, f) → shflForce_f += F(f←a) → S_f = F(f←b) + F(f←a)
T1: pair (b, g) → shflForce_g += F(g←b) → S_g = F(g←c) + F(g←b)
T2: pair (c, h) → shflForce_h += F(h←c) → S_h = F(h←d) + F(h←c)
T3: pair (d, e) → shflForce_e += F(e←d) → S_e = F(e←a) + F(e←d)
```

**SHUFFLE_WARP_DATA**（rotate-right-1）:

```
T0 ← T1: shflPosq = posq[g], shflForce = S_g = F(g←c) + F(g←b)
T1 ← T2: shflPosq = posq[h], shflForce = S_h = F(h←d) + F(h←c)
T2 ← T3: shflPosq = posq[e], shflForce = S_e = F(e←a) + F(e←d)
T3 ← T0: shflPosq = posq[f], shflForce = S_f = F(f←b) + F(f←a)

tj 更新: T0: tj=2, T1: tj=3, T2: tj=0, T3: tj=1
```

**Iteration j=1 结束后的状态**:

```
Thread | shflPosq | shflForce              | tj | 属于
-------|----------|------------------------|----|------
T0     | posq[g]  | F(g←c) + F(g←b)       | 2  | g
T1     | posq[h]  | F(h←d) + F(h←c)       | 3  | h
T2     | posq[e]  | F(e←a) + F(e←d)       | 0  | e
T3     | posq[f]  | F(f←b) + F(f←a)       | 1  | f
```

##### Iteration j=2

**计算**:

```
T0: pair (a, g) → S_g = F(g←c) + F(g←b) + F(g←a)
T1: pair (b, h) → S_h = F(h←d) + F(h←c) + F(h←b)
T2: pair (c, e) → S_e = F(e←a) + F(e←d) + F(e←c)
T3: pair (d, f) → S_f = F(f←b) + F(f←a) + F(f←d)
```

**SHUFFLE_WARP_DATA**（rotate-right-1）:

```
T0 ← T1: shflPosq = posq[h], shflForce = S_h = F(h←d) + F(h←c) + F(h←b)
T1 ← T2: shflPosq = posq[e], shflForce = S_e = F(e←a) + F(e←d) + F(e←c)
T2 ← T3: shflPosq = posq[f], shflForce = S_f = F(f←b) + F(f←a) + F(f←d)
T3 ← T0: shflPosq = posq[g], shflForce = S_g = F(g←c) + F(g←b) + F(g←a)

tj 更新: T0: tj=3, T1: tj=0, T2: tj=1, T3: tj=2
```

**Iteration j=2 结束后的状态**:

```
Thread | shflPosq | shflForce                         | tj | 属于
-------|----------|-----------------------------------|----|------
T0     | posq[h]  | F(h←d) + F(h←c) + F(h←b)        | 3  | h
T1     | posq[e]  | F(e←a) + F(e←d) + F(e←c)        | 0  | e
T2     | posq[f]  | F(f←b) + F(f←a) + F(f←d)        | 1  | f
T3     | posq[g]  | F(g←c) + F(g←b) + F(g←a)        | 2  | g
```

##### Iteration j=3 (最后一步)

**计算**:

```
T0: pair (a, h) → S_h = F(h←d) + F(h←c) + F(h←b) + F(h←a) = e 的总受力!
T1: pair (b, e) → S_e = F(e←a) + F(e←d) + F(e←c) + F(e←b) = e 的总受力!
T2: pair (c, f) → S_f = F(f←b) + F(f←a) + F(f←d) + F(f←c) = f 的总受力!
T3: pair (d, g) → S_g = F(g←c) + F(g←b) + F(g←a) + F(g←d) = g 的总受力!
```

**SHUFFLE_WARP_DATA**（rotate-right-1，最后一次）:

```
T0 ← T1: shflPosq = posq[e], shflForce = S_e = F(e←a) + F(e←d) + F(e←c) + F(e←b)
T1 ← T2: shflPosq = posq[f], shflForce = S_f = F(f←b) + F(f←a) + F(f←d) + F(f←c)
T2 ← T3: shflPosq = posq[g], shflForce = S_g = F(g←c) + F(g←b) + F(g←a) + F(g←d)
T3 ← T0: shflPosq = posq[h], shflForce = S_h = F(h←d) + F(h←c) + F(h←b) + F(h←a)
```

**循环结束后的最终状态**:

```
Thread | shflForce                                         | 初始 j-atom | 是否回到原位?
-------|---------------------------------------------------|-------------|-------------
T0     | F(e←a) + F(e←d) + F(e←c) + F(e←b) = F_total(e) | e           | ✓ (T0 初始持有 e)
T1     | F(f←b) + F(f←a) + F(f←d) + F(f←c) = F_total(f) | f           | ✓ (T1 初始持有 f)
T2     | F(g←c) + F(g←b) + F(g←a) + F(g←d) = F_total(g) | g           | ✓ (T2 初始持有 g)
T3     | F(h←d) + F(h←c) + F(h←b) + F(h←a) = F_total(h) | h           | ✓ (T3 初始持有 h)
```

**4 次 rotate-right-by-1 后，每个 j-atom 的 shflForce 恰好回到最初持有它的 thread，
且已累积了所有 i-atom 对该 j-atom 的力。**

同时，每个 thread 的 `force`（i-force，不旋转）也累积完毕：

```
T0: force_a = F(a←e) + F(a←f) + F(a←g) + F(a←h) = F_total(a)
T1: force_b = F(b←f) + F(b←g) + F(b←h) + F(b←e) = F_total(b)
T2: force_c = F(c←g) + F(c←h) + F(c←e) + F(c←f) = F_total(c)
T3: force_d = F(d←h) + F(d←e) + F(d←f) + F(d←g) = F_total(d)
```

**写回**:

```cuda
// j-force: offset = y * TILE_SIZE + tgx, 即线程的永久 lane index
atomicAdd(&forceBuffers[y*4 + 0], ...shflForce_T0...);  // → atom e 的总受力
atomicAdd(&forceBuffers[y*4 + 1], ...shflForce_T1...);  // → atom f 的总受力
atomicAdd(&forceBuffers[y*4 + 2], ...shflForce_T2...);  // → atom g 的总受力
atomicAdd(&forceBuffers[y*4 + 3], ...shflForce_T3...);  // → atom h 的总受力

// i-force: offset = x * TILE_SIZE + tgx
atomicAdd(&forceBuffers[x*4 + 0], ...force_T0...);      // → atom a 的总受力
atomicAdd(&forceBuffers[x*4 + 1], ...force_T1...);      // → atom b 的总受力
atomicAdd(&forceBuffers[x*4 + 2], ...force_T2...);      // → atom c 的总受力
atomicAdd(&forceBuffers[x*4 + 3], ...force_T3...);      // → atom d 的总受力
```

---

#### 32-atom tile 的一般性证明

**定理**: 在 TILE_SIZE=N 的非对角线 tile 中，N 次 rotate-right-by-1 后，thread `i`
的 `shflForce` 恰好是 j-atom `y*N + i` 受到的来自 block_x 中所有 N 个 atom 的总力。

**证明**:

1. **j-atom 数据的位移规律**: rotate-right-by-1 将 thread `k` 中的数据移到 thread `k-1 mod N`。
   经过 `s` 次旋转后，初始 thread `k` 的数据位于 thread `(k-s) mod N`。

2. **shflForce 的追踪**: 设 j-atom `J_i = y*N + i`（thread `i` 的初始 j-atom）。
   在 iteration `s`（0 ≤ s < N）**计算之前**，`J_i` 的 shflForce 经历了 `s` 次 SHUFFLE，
   当前位于 thread `(i - s) mod N`。

3. **每次迭代的累加**: iteration `s` 中，持有 `J_i` 的 shflForce 的 thread 是
   `T_{(i-s) mod N}`，其 i-atom 为 `x*N + (i-s) mod N`。
   该 thread 计算 pair (`x*N+(i-s) mod N`, `J_i`)，并将力累加到 `J_i` 的 shflForce 上。

4. **完整累积**: s 从 0 到 N-1，`J_i` 的 shflForce 被以下 i-atom 的贡献累加：
   - s=0: i-atom `x*N + i`
   - s=1: i-atom `x*N + (i-1) mod N`
   - ...
   - s=N-1: i-atom `x*N + (i-N+1) mod N`

   这恰好覆盖 block_x 中所有 N 个 atom 各一次。因此：
   ```
   shflForce_Ji = Σ_{k=0}^{N-1} F(x*N + (i-k) mod N → J_i)
                = J_i 受到来自 block_x 所有 atom 的总力
   ```

5. **回到原位**: N 次旋转后，`J_i` 的数据位于 thread `(i-N) mod N = i`。
   恰好回到初始 thread `i`。写回地址 `y*N + tgx = y*N + i` 正确指向 `J_i`。

---

#### 实际代码 (openmm/openmm `nonbonded.cu`)

```cuda
// ===== 非对角线 tile (x != y) 核心循环 =====
unsigned int j = y * TILE_SIZE + tgx;  // 每个 thread 从 global 读 1 个 j-atom
real4 shflPosq = posq[j];
real3 shflForce;                        // 初始化为零
shflForce.x = 0.0f;
shflForce.y = 0.0f;
shflForce.z = 0.0f;
DECLARE_LOCAL_PARAMETERS               // 如 real2 shflSigmaEpsilon;
LOAD_LOCAL_PARAMETERS_FROM_GLOBAL      // 从 global 加载 j-atom 参数
unsigned int tj = tgx;                  // 初始 tj = lane index

for (j = 0; j < TILE_SIZE; j++) {
    real4 posq2 = shflPosq;
    real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
    // ... PBC, r² 计算 ...
    LOAD_ATOM2_PARAMETERS               // sigmaEpsilon2 = shflSigmaEpsilon;
    atom2 = y*TILE_SIZE + tj;           // j-atom 全局 index

    real dEdR = 0.0f;
    const real interactionScale = 1.0f; // 非对角线: 不折半
    COMPUTE_INTERACTION                 // 用户定义的力计算代码

    // 力累积:
    delta *= dEdR;
    force.x -= delta.x;                // i-force (register, 不旋转)
    force.y -= delta.y;
    force.z -= delta.z;
    shflForce.x += delta.x;            // j-force (跟着旋转!)
    shflForce.y += delta.y;
    shflForce.z += delta.z;

    SHUFFLE_WARP_DATA                   // shflPosq + shflForce + params 一起 rotate-right-1
    tj = (tj + 1) & (TILE_SIZE - 1);   // cyclic: 0→1→2→...→31→0
}

// ===== j-force 写回 (非对角线) =====
// offset 基于 tgx (永久 lane index), 不是 tj (循环变量)
// 32 步旋转后 shflForce 恰好回到初始 j-atom 对应的 thread
const unsigned int offset = y * TILE_SIZE + tgx;
atomicAdd(&forceBuffers[offset], realToFixedPoint(shflForce.x));
atomicAdd(&forceBuffers[offset + PADDED_NUM_ATOMS], realToFixedPoint(shflForce.y));
atomicAdd(&forceBuffers[offset + 2*PADDED_NUM_ATOMS], realToFixedPoint(shflForce.z));

// ===== i-force 写回 (对角线 + 非对角线共用) =====
const unsigned int offset_i = x * TILE_SIZE + tgx;
atomicAdd(&forceBuffers[offset_i], realToFixedPoint(force.x));
atomicAdd(&forceBuffers[offset_i + PADDED_NUM_ATOMS], realToFixedPoint(force.y));
atomicAdd(&forceBuffers[offset_i + 2*PADDED_NUM_ATOMS], realToFixedPoint(force.z));
```

#### OpenMM 方案汇总

```
tile = 32×32 = 1024 pairs (非对角线):
  i-force: register 累积 32 步, 不旋转 → 3 次 atomicAdd / i-atom
  j-force: register 累积 + rotate-right-1 旋转 32 步 → 3 次 atomicAdd / j-atom
  总 atomicAdd: (32 i-atoms + 32 j-atoms) × 3 = 192 次 / 1024 pairs
  → 0.19 次 atomicAdd / pair
  显式 reduction 步骤: 0 (归约隐含在循环旋转中)

对角线 tile (x == y):
  只累积 i-force (无 shflForce), interactionScale = 0.5
  总 atomicAdd: 32 i-atoms × 3 = 96 次 / 1024 pairs
  → 0.09 次 atomicAdd / pair
```

---

### 三方案定量对比

| | 朴素 atomicAdd | GROMACS `shfl_down` | OpenMM rotate-right-1 |
|--|---------------|--------------------|----------------------|
| 计算 unit | 64-1024 pairs | 64 pairs (8×8 cluster) | 1024 pairs (32×32 tile) |
| i-force 累积 | 每 pair 1 次 atomic | register + 最终 1 次 atomic | register + 最终 1 次 atomic |
| j-force 归约 | 每 pair 1 次 atomic | `shfl_down` × log₂(N) 轮 + 1 次 atomic | 隐含在 N 步 rotate-right-1 + 1 次 atomic |
| 显式 reduction 步骤 | 0 | log₂(cluster_size) 步 | **0** |
| atomicAdd / pair | 6.0 | 0.75 | **0.19** |
| shuffle 指令 | 无 | `__shfl_down(var, delta)` | `__shfl_sync(var, tgx+1)` |
| shuffle 语义 | - | 位移求和: lane_i += lane_{i+delta} | rotate-right-1: T_i ← T_{(i+1) mod N} |
| shflForce 归约时机 | 不归约 (直接 atomic) | 循环后显式 log₂(N) 步 | 循环中隐式累积 (零额外开销) |
| 适用分组粒度 | 任意 | 固定 cluster size (8) | 固定 tile size = warp size (32) |

### 核心区别总结

**GROMACS** 的 `__shfl_down` 是 **"先算完再归约"**：
- 所有 thread 各自计算完 cluster pair 内所有 i-cluster 的力
- 循环结束后，显式调用 log₂(N) 步 `shfl_down` 把 N 个 thread 的部分和合并为 1 个
- 归约是一个**独立阶段**，需要额外指令和时间

**OpenMM** 的 rotate-right-1 是 **"边算边归约"**：
- j-atom 的坐标、参数、**已累积的力**作为一组数据，在每次迭代末尾 rotate-right-1
- 每到一个新 thread，该 thread 立即累加自己的 i-atom 对该 j-atom 的力贡献
- N 步后数据回到原位时，shflForce 已自动收集了所有 N 个 i-atom 的贡献
- **零额外归约步骤**: SHUFFLE_WARP_DATA 是循环体的一部分，与力计算串行但不额外增加循环次数

---

## Optimization Roadmap

### Phase 1: Kernel 内层优化 (高优先级)

**目标**: 降低 atomicAdd 开销，提升 register 利用率

#### 1.1 实现对角线循环 + shuffle 旋转 force reduction

替换当前的朴素 `atomicAdd` per pair 为 OpenMM 风格的对角线循环。

Kernel 结构变更:

```
当前:                              → 改为:
256 threads / block                256 threads / block (8 warps)
每 thread 串行 4 pairs             每 warp 处理 1 tile (对角线循环)
6 次 atomicAdd / pair              6 次 atomicAdd / atom (32 步循环结束后)
共 ~768 atomicAdd / tile           共 192 atomicAdd / tile (1024 pairs)
```

需要实现的核心宏/函数:
- `SHUFFLE_WARP_DATA`: 用 `__shfl_sync()` 旋转 posq + parameters + shflForce
- 对角线 tile index 计算: `tj = (tgx + j) & (TILE_SIZE - 1)`
- 对角线 tile (x == y) 的 interactionScale = 0.5 (避免重复计算)
- 非对角线 tile (x != y) 的 interactionScale = 1.0

参考: `openmm/openmm` 仓库 `platforms/cuda/src/kernels/nonbonded.cu` 的
`computeNonbonded` kernel。

#### 1.2 Shared memory 优化

当前已使用 shared memory 预加载 i-tile 和 j-tile 坐标。
改为对角线循环后只需预加载 i-tile (j-atom 通过 shuffle 旋转获得):

```
__shared__ float smem_pos_i[32 * 3];  // i-tile 坐标 (预加载)
// j-tile 坐标通过 shuffle 旋转，不需要 shared memory
```

#### 1.3 Exclusion bitmask 处理优化

在 tile pair 列表中分离两类:
- **含 exclusion 的 tile**: 用对角线循环 + per-step exclusion check
- **无 exclusion 的 tile**: 跳过 exclusion 检查，更快路径

参考 GROMACS 的 j-entry 排序策略: 将有 exclusion 的排在前面，无 exclusion 的走 fast path。

### Phase 2: Neighbor List 构建 GPU 化 (中优先级)

**目标**: 将 tile pair 搜索从 CPU O(N^2) 迁移到 GPU

#### 2.1 GPU bounding box 计算

当前 `_build_interaction_tiles` 在 CPU 上遍历所有 tile pair (O(N_tiles^2))，
检查 bounding box 距离 < build_radius。

迁移到 GPU kernel:
- `findBlockBounds` kernel: 计算 each tile 的 center + half_width
- `findBlocksWithInteractions` kernel: tile pair bounding box 检查
- 用 warp vote (`__ballot_sync`) 生成 interaction mask

参考: OpenMM `platforms/cuda/src/kernels/findInteractingBlocks.cu`

#### 2.2 Padded cutoff + 延迟重建

当前已实现 `skin / 2` 位移检测 + rebuild。保持此策略，与 OpenMM 一致。

### Phase 3: Electrostatics 扩展 (低优先级)

**目标**: 支持长程静电方法

#### 3.1 Reaction Field

最简单的扩展。在 Coulomb direct 项上增加 RF 修正:
```
V = q_i * q_j * (1/r + k_rf * r^2 - c_rf)
F = q_i * q_j * (1/r^2 - 2 * k_rf * r)
```

参数: `k_rf`, `c_rf` 由溶剂介电常数和 cutoff 计算。

#### 3.2 PME 直接空间修正

在 kernel 内部增加 Ewald screening 修正:
```
// 解析修正 (推荐，精度足够):
float pme_corr = pmeCorrF(beta^2 * r^2);  // Pade 有理逼近
fInvR += qi * qj * (pairExclMask * invR2 * invR + pme_corr * beta^3);

// Pade 近似系数 (from GROMACS nbnxm_hip_kernel_body.h:262-296):
// 分子: FN6..FN0, 分母: FD4..FD0
// 单精度相对误差 < 1e-7
```

PME 倒空间 (FFT-based) 作为独立模块，不在 nonbonded kernel 内。

#### 3.3 LJ Potential Switch

5 阶多项式平滑截断 (C1 连续):
```cuda
if (r > r_switch) {
    float x = r - r_switch;
    float sw = 1 + x*x*x * (C3 + x*(C4 + x*C5));
    float dsw = x*x * (3*C3 + x*(4*C4 + x*5*C5));
    fInvR = fInvR * sw - rInv * eLJ * dsw;
    eLJ *= sw;
}
```

系数 C3/C4/C5 由 switch_distance 和 cutoff_distance 预计算，满足:
- S(r_switch) = 1, S'(r_switch) = 0
- S(r_cutoff) = 0, S'(r_cutoff) = 0

参考: GROMACS `nbnxm_hip_kernel_body.h:228-259`

### Phase 4: 大规模优化 (远期)

仅在体系 > 50 万原子时考虑:

- GPU-side pairlist sorting (workload 均衡化)
- Dynamic pair-list pruning (inner/outer cutoff)
- Twin cutoff (rcoul != rvdw) for PME load balancing
- LJ-Ewald grid correction

---

## 从 GROMACS 可借鉴的具体技巧

### 值得借鉴

| 技巧 | 来源文件 | 何时引入 |
|------|---------|---------|
| Register 累积 i-force + 一次 atomic | `nbnxm_hip_kernel_body.h:676-680` | Phase 1 |
| `__shfl_down` j-force 归约 | `nbnxm_hip_kernel_body.h:331-355` | Phase 1 (备选方案) |
| PME 解析修正 Pade 近似 | `nbnxm_hip_kernel_body.h:262-296` | Phase 3.2 |
| LJ potential switch 多项式 | `nbnxm_hip_kernel_body.h:228-259` | Phase 3.3 |
| LJ force switch 多项式 | `nbnxm_hip_kernel_body.h:137-166` | Phase 3.3 (备选) |
| Exclusion bitmask 排序 (fast path) | `pairlist.h` j-entry sorting | Phase 1.3 |
| `__launch_bounds__` 占用率控制 | `nbnxm_hip_kernel_body.h:573` | Phase 1 |

### 不值得引入 (对 mdpy 不适用)

| 技术 | 原因 |
|------|------|
| 8×8 cluster pair-list | 32-atom tile 更适合 mdpy 的体系和实现 |
| 49+ kernel 变体编译期特化 | CuPy RawKernel 不适合此模式，运行时参数更简单 |
| Super-cluster / 2×2×2 grid | 需要 MPI domain decomposition 才有意义 |
| Dynamic pruning (inner/outer cutoff) | 过度复杂，padded cutoff 足够 |
| GPU-side pairlist histogram sorting | 仅在 > 50 万原子时有收益 |
| Twin cutoff | 需要 PME + 负载均衡才需要 |
| Soft-core FEP kernel | mdpy 当前不涉及自由能计算 |

---

## Key Reference Files

### GROMACS (本地: /home/ubuntu/Programs/gromacs-2026.2/)

| 文件 | 内容 |
|------|------|
| `src/gromacs/nbnxm/hip/nbnxm_hip_kernel_body.h` | HIP GPU kernel 完整实现 (force 计算 + reduction) |
| `src/gromacs/nbnxm/gpu_types_common.h` | GPU 数据结构 (NBAtomDataGpu, NBParamGpu, GpuPairlist) |
| `src/gromacs/nbnxm/nbnxm_enums.h` | ElecType / VdwType 枚举 + cluster size 常量 |
| `src/gromacs/nbnxm/pairlist.h` | CPU/GPU pairlist 数据结构 (bitmask encoding) |
| `src/gromacs/nbnxm/grid.h` | 空间网格构建 |

### OpenMM (GitHub: openmm/openmm)

| 文件 | 内容 |
|------|------|
| `platforms/cuda/src/kernels/nonbonded.cu` | CUDA nonbonded kernel (对角线循环 + shuffle 旋转) |
| `platforms/cuda/src/kernels/findInteractingBlocks.cu` | GPU neighbor list 构建 |
| `platforms/cuda/src/CudaNonbondedUtilities.cpp` | Tile 管理 + 数据传输 |
| `platforms/cuda/src/CudaNonbondedUtilities.h` | GPU 数据结构 (interactingTiles, blockCenter 等) |
| `include/openmm/reference/ReferenceNeighborList.h` | CPU voxel hash neighbor list |
