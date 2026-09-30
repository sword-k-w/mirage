**MPK 阶段 5 导读 RMSNorm 任务内部的 CUDA 实现**

承接 [阶段 4](stage-4-guide-zh.md)，我们已经知道什么时候调用 `_execute_task()`。这次只看其中一个 RMSNorm device 函数怎样搬数据、归约和写回。基于 `97de2592`，建议 3～4 小时；主线以 Hopper 分支、BF16、每任务一行、H=4096 为例。

主文件：[rmsnorm_hopper.cuh](../../include/mirage/persistent_kernel/tasks/hopper/rmsnorm_hopper.cuh)。配套 [copy_sm80.cuh](../../include/mirage/persistent_kernel/tasks/common/copy_sm80.cuh)、[utils.cuh](../../include/mirage/persistent_kernel/tasks/common/utils.cuh)、[task_register.cc](../../src/kernel/task_register.cc)。架构名字不等于独立运行入口，应先用阶段 2 的注册链确认实际选择。

**1．写下 device 函数的输入契约**

```cpp
template <typename T, int BATCH_SIZE, int HIDDEN_DIM, int NUM_THREADS = 256>
__device__ __forceinline__ void rms_norm_hopper_impl(
    void const *input_ptr,
    void const *weight_ptr,
    void *output_ptr,
    float eps);
```

这是一份简化声明。`input_ptr/output_ptr` 已经定位到当前 task 的 tile，`BATCH_SIZE` 是 tile 内行数。阶段 1 的 16 行切成 16 项任务，因此本例 BATCH_SIZE=1；NUM_THREADS 默认 256。

先列契约，再看性能技巧：输入和输出按函数使用的连续行布局组织；权重有 H 个元素；共享内存和参加协作的线程配置要满足实现要求；所有参与线程必须按设计到达同步点。模板能实例化并不意味着任意指针、stride 或 block 形状都适用。

它是 `__device__`，由常驻 worker 调用，不为每一行再 launch 一个新 kernel。函数里的 `threadIdx.x` 来自当前 worker CTA，不能把原始算子 grid 的 tile 编号简单等同于这里的 `blockIdx.x`。

**2．先手算编译期常量**

沿定义逐个代入 T=BF16、H=4096、线程数=256：

| 常量 | 本例数值 | 含义 |
|---|---:|---|
| `EVEN_SPLIT` | true | H 能被线程数整除 |
| `BYTES_PER_THREAD` | 32 | H/256 个 BF16 所占字节 |
| `BYTES_PER_CP` | 16 | 本次选择的单次异步拷贝宽度 |
| `CHUNK_SIZE` | 8 | 一次拷贝对应 8 个 BF16 元素 |
| `TILE_SIZE` | 2048 | 256 线程各搬 8 个元素 |
| `NUM_TILES` | 2 | 覆盖一行需要两批拷贝 |
| `NUM_CHUNKS_OUTPUT` | 512 | 写回整行的向量块数量 |
| `NUM_WARPS` | 8 | 256/32 |

这里的“拷贝 tile”与“MPK task 的 tensor tile”是不同层次：一项 task 已经拿到一行，而这一行内部又分两批异步搬运。

练习：只用纸笔计算这些值，尤其检查单位是字节还是元素。不要从 `BYTES_PER_THREAD=32` 推导每次 cp.async 都搬 32 字节，当前辅助函数支持的是 4/8/16 字节宽度。

**3．画 shared memory 布局**

函数用 `extern __shared__ char smem[]`，在里面依次划分输入、权重、输出和归约区：

```text
byte 0      ... 8191    input，4096 个 BF16
byte 8192   ... 16383   weight，4096 个 BF16
byte 16384  ... 24575   output，4096 个 BF16
byte 24576  ...         reduce_smem，存放 warp 局部和
```

归约需要至少容纳 8 个 float，即 32 字节，因此这个例子上述使用区域为 24,608 字节。它不等于整个 runtime 的 shared memory 预算：worker 还有自身控制数据，启动时还可能为其他算子预留更大的动态共享内存。

为什么输出先写 shared memory？后面使用向量化写回；计算线程逐元素负责的下标，与负责某个连续向量块写出的线程不一定相同，因此中间 CTA 同步很重要。

**4．看清异步拷贝的两批数据**

thread t 首先搬 `t*8` 开始的 8 个输入元素和对应权重；256 个线程覆盖前 2048 个元素。然后提交这一组异步拷贝。

循环第一次又提交后 2048 个元素，执行 `cp_async_wait<1>()`，再同步 CTA，读取前一批进行平方求和。循环最后执行 `cp_async_wait<0>()`，保证剩余拷贝完成，再处理后一批。

```text
提交组 0：input/weight 的前半行
提交组 1：input/weight 的后半行
wait<1>：允许最新一组仍未完成，确保较早组已完成
计算前半行
wait<0>：等剩余组全部完成
计算后半行
```

这描述的是提交/等待关系，不保证硬件时间线上必然出现某个固定比例的重叠。`copy_sm80.cuh` 中名为 `cp_async_fence()` 的函数实际发出 `cp.async.commit_group`，不是跨 CTA 的通用 memory fence；`wait<0>` 使用 `cp.async.wait_all`，其他 N 使用 `wait_group N`。N 表示允许仍未完成的最近 group 数，不是“等待 N 纳秒”或“等待第 N 个 tile”。[NVIDIA PTX cp.async 说明](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group)

`wait` 解决当前线程异步拷贝的完成关系，`__syncthreads()` 再协调 CTA 的其他线程。因为后续线程会读取不止自己搬来的值，不能只保留其中一个等待动作。

**5．从每线程局部和到整行的和**

局部计算循环使用 `i=threadIdx.x; i<tile_len; i+=NUM_THREADS`。本例 thread 0 在第一批读取下标 `0,256,...,1792`，第二批读取 `2048,2304,...,3840`，共 16 个元素。thread 1 同样处理每个位置加 1 的序列。

注意搬运时 thread 0 负责连续块 `[0,8)`，计算时却按步长 256 取值。这不矛盾：数据已经进入 block 共享存储，计算阶段可以重新分配工作。

归约分两级：

1. 每个 warp 用 XOR shuffle，按偏移 `16,8,4,2,1` 合并 32 个局部和。
2. 每个 warp 的 lane 0 将结果写到 `reduce_smem[warp_id]`；CTA 同步后，前 8 个线程读入这些值，再用 `4,2,1` 合并。

最后 thread 0 将整行平方和写到 `reduce_smem[0]`，同步后各线程读取，用 `rsqrt(sum/H + eps)` 得到归一化因子。shuffle 只在 warp 内交换寄存器值；跨 warp 的阶段依赖 shared memory 与同步。

练习：把每个 warp 想成一个数 `s0..s7`，手算第一 warp 的前 8 个 lane 在第二级归约如何得到总和。其他 warp 不需要提供跨 warp shuffle。

**6．输出计算与向量化写回**

线程按步长遍历 H 个元素，将 input、weight 转成 float，计算归一化与乘权重，再转回 T，写入 shared output。CTA 同步后，按 `BYTES_PER_CP` 的宽度将连续数据块写到 global output。

本例输出有 512 个 16-byte 块，每线程负责两个；计算和写回的线程分工可以不同。读取地址已经是当前任务输出行的起点，所以不再加原始全图 task 编号。

最终数据进入 global memory，供图中后继 worker 访问。返回 worker 后，阶段 4 的 CTA 同步和完成计数发布才继续推进跨任务依赖。算子内部同步与 runtime 依赖发布各负责一层。

**7．再读尾部处理与数值边界**

第二遍关注 `EVEN_SPLIT`、`off < HIDDEN_DIM`、`tile_len` 和向量块整除断言。可以手算 H=4104、线程数仍 256：BYTES_PER_CP 仍为 16，CHUNK_SIZE=8，NUM_TILES=3，最后一批只有 8 个有效元素。这是索引练习，不是声明该 shape 在整条工具链上已运行通过。

对照 [test_rmsnorm_uneven_testmode.py](../../tests/runtime_python/test_mode/test_rmsnorm_uneven_testmode.py)，查看它实际覆盖哪些形状和检查条件。理解边界处理后再考虑运行，不能把“已有尾部判断”扩大成任意 shape 都支持。

数值方面回看阶段 1：reference 默认 eps 为 `1e-5`，注册实现用 `1e-6`；平方和在 float 中累加，树形归约与参考运算顺序可能不同，输出回到 BF16。误差要结合 eps、累加顺序、输入尺度和 dtype 解释。

为了理解 eps 的作用，可手算一行全零输入：均方值为零，eps 让倒平方根有限，乘零得到零。只用随机大尺度输入测试，未必能突出 eps 差异。

**8．自测和可选的下一种算子**

| 自测题 | 核对要点 |
|---|---|
| 本例 BATCH_SIZE 是 16 还是 1？ | 每任务一行时为 1，16 是整图输入行数 |
| H=4096 要提交多少批拷贝？ | 按当前常量为 2 批，每批 2048 元素 |
| 搬运线程与计算线程必须负责同一元素吗？ | 不必，shared memory 和同步允许重新分工 |
| `wait<1>` 为什么能读取前半行？ | 较早提交组已经完成，最多保留最近一组未完成 |
| 为什么要两级归约？ | shuffle 在 warp 内，跨 warp 需共享存储和同步 |
| 输出 shared buffer 同步能删吗？ | 不能只凭顺序代码删，其他线程可能负责向量写回 |
| 函数返回是否已通知后继任务？ | 通知由 worker runtime 的后续完成协议承担 |

交付笔记是一张 shared memory 布局、一张拷贝组时间图、一张两级归约图。可选扩展再读 SiLU-Mul；想读矩阵乘法时只选当前架构的一种实现，按“形状→线程分工→搬运→MMA→写回”分开学习。TMA 可参考 [tma.md](tma.md)，无需把它作为本阶段的前置条件。

下一阶段将已理解的机制连回模型：[阶段 6](stage-6-guide-zh.md)。
