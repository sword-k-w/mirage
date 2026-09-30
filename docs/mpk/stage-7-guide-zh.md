**MPK 阶段 7 导读 多 GPU 的计算通信和远程依赖**

承接 [阶段 6](stage-6-guide-zh.md)。本阶段用两个 GPU 的张量并行例子，追踪 all-reduce 的任务注册、数据搬运和完成信号。基于 `97de2592`，建议 3～4 小时；没有多卡也能通过手算和源码完成。

当前仓库的多卡相关代码有启动路径限制与依赖版本要求，下文会具体指出。本阶段不提供未经验证的完整多卡启动配方，也没有运行多卡实验。

**1．先用一个矩阵乘法理解为什么要通信**

PyTorch 常见线性层约定 `Y=X @ W.T`。若沿输出维度划分 W 的行，各 GPU 得到一部分输出列；若沿输入/归约维划分 W 的列，各 GPU 得到完整输出形状的一部分和。

用后者举例，假设输入 `[1,2,3,4]`，输出只有一个值，权重 `[10,20,30,40]`：

```text
GPU 0：1*10 + 2*20 = 50
GPU 1：3*30 + 4*40 = 250
完整结果：50 + 250 = 300
```

all-reduce(sum) 让参与者都得到 300；all-gather 会让参与者拿到 `[50,250]`，之后还需本地相加。两者不是同一种数学操作。

对于 Qwen3 MLP，gate/up 的输出特征可在 rank 间切分，down projection 则会产生要相加的局部贡献。回到 builder 的 `world_size>1` 分支，找 attention output projection 与 MLP down 后面的 `allreduce_layer()`。

残差也要核对：如果每个 rank 在局部结果里都加入同一份 residual，sum 后会重复加 world_size 次。实际实现有 rank 条件等处理，按所选 `linear_with_residual` 注册路径核对；不要将单卡残差公式直接逐卡复制。

**2．Python 实际选择哪一种 collective**

读 [multigpu.py](../../python/mirage/mpk/multigpu.py) 的 `get_collective_capabilities()`、`auto_select_allreduce_implementation()`，再读 [PersistentKernel.allreduce_layer](../../python/mirage/mpk/persistent_kernel.py)。

输入/输出是 `[batched_tokens, hidden]`，中间 buffer 描述为 `[world_size, batched_tokens, hidden]`。选择策略按硬件能力判断：当前代码在 target_cc≥90 且 VMM、multicast、peer access 条件满足时选 tile allreduce，否则退回 allgather+local reduction。

这个 `auto_select` 是规则选择，不是在运行时实测所有算法再选最快。也不能因文件定义了 `AllGatherStrategy` 等基类，就推断每种 collective 都已实现；某些选择函数仍抛 `NotImplementedError`。

| 分支 | 构图时添加什么 |
|---|---|
| `AllReduceStrategy_AllgatherReduce` | `nvshmem_allgather_strided_put`，随后 `reduction` |
| `AllReduceStrategy_NvshmemTile` | 一类 `nvshmem_tile_allreduce` 任务，并安排 team 资源 |

第一遍先追 fallback，因为数据和信号关系更容易看清；然后再对照 tile 分支，不要将两个分支的 event 机制混写。

**3．NVSHMEM 的地址与 PE**

PE 可以先理解为参与 NVSHMEM 程序的一个处理端，在这条典型部署路径上与 rank/GPU 对应。symmetric memory 是参与者按 NVSHMEM 规则建立、供远程操作寻址的内存对象；不能把一个普通 `cudaMalloc` 指针随意传给远端，当作已满足这些要求。

NVSHMEM 支持从 device 发起远程 put/get；顺序、完成和消费方观察仍需要协议，API 返回不一定等价于远端数据已可消费。[NVSHMEM 使用说明](https://docs.nvidia.com/nvshmem/api/latest/using.html)

在本仓库追三个位置：

- `PersistentKernel.new_tensor(io_category="nvshmem_tensor")` 注册相应分配类别。
- `runtime.cc` 的 IO 描述生成会为相关类别发出 `nvshmem_malloc`。
- `persistent_kernel.cuh` 的 `gpu_malloc()` 在 `USE_NVSHMEM` 下使用 NVSHMEM 分配 runtime 数据。

不同策略要求远程访问的对象不尽相同。给自己的图标注：发送源、接收 buffer、远程 signal counter、tile collective 的输入/输出分别由谁分配。不要看到某个 demo 用 `attach_input(torch_tensor)`，就推断它适用于所有通信分支。

**4．跟踪 fallback 的一块数据**

源码入口是 [tasks/ampere/allreduce.cuh](../../include/mirage/persistent_kernel/tasks/ampere/allreduce.cuh) 的 `nvshmem_allgather_strided_put()`。虽然路径名含 ampere，这个注册分支应按调用链判断，不按目录名推断实际 GPU。

单项通信任务大致完成：

```text
对每个 active token：
  把当前 tile 的数据 put 到目标 rank 的 buffer 对应区域
等待相关非阻塞操作完成并同步 CTA
thread 0 将目标 rank 的 signal counter 加 1
```

实际顺序是循环 `nvshmemx_putmem_nbi_block`，随后 `nvshmem_quiet()`、`__syncthreads()`，再由 thread 0 调 `nvshmemx_signal_op(..., NVSHMEM_SIGNAL_ADD, target_gpu_id)`。对非阻塞远程更新，消费者仍须通过等待/测试等同步观察结果；不能仅在本地看见“已发起 put”。quiet 的完成保证也要结合发起端和参与线程语义理解。[NVSHMEM ordering 说明](https://docs.nvidia.com/nvshmem/api/latest/using.html)

为什么逐行 put？这个 tile 沿 hidden 维切分后，每行有效输出长度可能小于原始行跨度；跨行并不一定是一整块连续内存。注册函数特意检查 `input_map` 并传入 stride，循环按行定位。

练习：设 B=2，hidden=8，每 task 处理 4 个特征。画出一项 task 在两行中的有效区域，说明为何不能直接把“2×4 元素”从原始行主序张量起点连续拷走。

**5．信号怎样变成下游的依赖**

在 [task_register.cc](../../src/kernel/task_register.cc) 找 `register_nvshmem_allgather_strided_put_task()`：它从 `trigger_event` 取出目标 GPU 与 event 位置，将 runtime counter 地址传给通信函数。

再读 [runtime.cc](../../src/kernel/runtime.cc) 的 `get_num_subtasks()`：这种 allgather put 每个 tile 对其他 `world_size-1` 个 rank 展开子任务。4 卡时，一个 tile 有 3 个远端目标；不是把数据发送给自己四次。

接收方的 reduction task 在执行前通过带 NVSHMEM 标记的 `dependent_event` 等待，worker 调用 `nvshmem_signal_wait_until`。本地普通等待用“至少达到门槛”，这里代码采用 NVSHMEM equality 比较；理解时保留这个差别，不能擅自统一成一段伪代码。

```text
GPU 0 局部计算 → put GPU 1 接收区 → 完成/信号
                                           ↓
GPU 1 本地输入 + 来自 GPU 0 的接收数据 → 等待满足 → reduction
```

本地 reduction 对本 rank 的贡献直接读 input，对其他 rank 的贡献读 buffer 中对应槽位。因而 allgather buffer 自己那一槽不一定需要按同一路径写入。

worker 的远程 trigger 分支注明通信任务内部已完成 signal，外层不再额外加一次。旁边也有“该分支不再使用”的 TODO；阅读时应沿实际策略、生成任务和宏判断，不能据这条注释认定 fallback 已被删除，也不能未经运行就断言每个环境都会走到它。

当前预派发后，一些远程等待直接由 worker 完成。虽然 runtime 保留 remote scheduler/第二 worker queue 的结构，不能把它讲成“所有跨卡等待都必须由 remote scheduler 轮询和转发”。

**6．tile allreduce 与 team**

读 [Hopper allreduce](../../include/mirage/persistent_kernel/tasks/hopper/allreduce.cuh)，先抓输入 tile layout、`active_tokens`、`teams[task_offset]`、collective 调用四个点。Python 的 `allocate_nvshmem_teams()` 和 runtime 中的 team split 负责准备所需资源。

用不同 task 的 team/offset 组织并发 collective，是需要与任务顺序一起考虑的协议；不能只把它当成替换一个函数名。Blackwell 的 [allreduce.cuh](../../include/mirage/persistent_kernel/tasks/blackwell/allreduce.cuh) 还有针对该路径的专用实现，第一遍无需全部展开。

Hopper 代码选择的算法名含 NBI，且显式 `tile_collective_wait_block` 调用被注释掉。应记录为需要结合所用 NVSHMEM 版本、算法完成语义和后续读取核验的位置。本导读不能仅凭命名就判定这里正确或错误，也不能把未执行的注释当成实际等待。

**7．当前版本的启动限制必须与设计分开**

在 `launch_persistent_kernel()` 中，源码明确注释拆分 worker/scheduler 路径不支持其 NVSHMEM collective launch 方式；单 persistent kernel 分支使用 `nvshmemx_collective_launch()`。但同一提交的初始化默认设置 `split_worker_scheduler=true`，没有在此处按 USE_NVSHMEM 自动切换。

这是本地可直接看到的路径不一致。因此不要拿单卡测试默认配置加上 `world_size=2` 就当作已准备好的多卡运行方法。真正运行前要确认进入受支持的 launch 路径、NVSHMEM 版本和 symmetric memory/team 配置；本次仅记录限制，不修改核心代码或声称多卡验证通过。

这不妨碍完成本阶段的阅读：我们已经能够指出需要计算什么、传什么、通知什么、在哪里等，以及启动配置为什么是正确性的一部分。

**8．手算一次两卡执行并自测**

回到 50 与 250 的例子，为每张卡列五项状态：本地贡献、远端接收区、signal counter、reduction 是否可执行、最终输出。让 GPU 1 的计算慢一拍，检查 GPU 0 是否会读到旧 buffer。

再扩展成每张卡两块独立 tile：只有某个 tile 的全部贡献到齐时才能消费该 tile；通信与另一个独立 tile 的计算可能重叠，是否实际重叠还受 worker 分配、协议和硬件制约。不要把“在同一个大 kernel”当作重叠证据。

| 自测题 | 核对要点 |
|---|---|
| all-gather 得到 `[50,250]` 是否等于 all-reduce？ | 还需相加，all-reduce 各 rank 得到 300 |
| 每个 rank 都先加 residual 会怎样？ | sum 可能重复累加，需要核对 rank 条件 |
| tile 逐行 put 为什么要 stride？ | 有效区域与原 tensor 行跨度不一定相等 |
| 数据到达与发送函数返回是否同一概念？ | 非阻塞路径不能这样假设，需完成与观察协议 |
| 4 卡一项 allgather tile 有几个远端子任务？ | 3 个，分别发给其他 rank |
| worker 是否再为远端 put 加一次 signal？ | 此路径由通信函数内部通知，不能重复计数 |
| 规则选择 tile 算法是否证明最快？ | 不是实测 autotuning |
| 能否直接沿当前 split 默认运行 NVSHMEM？ | 不能当作已验证配置，需先解决/核对启动路径限制 |

下一步读 [阶段 8](stage-8-guide-zh.md)，用可观测证据检验性能解释。
