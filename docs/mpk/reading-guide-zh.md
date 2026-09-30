MPK 核心实现分阶段导读
====================

适用背景：会基础 C++ / CUDA，对 LLM inference 有简单了解；不要求先掌握编译器、CUTLASS 或分布式系统。

本计划基于本地提交 `97de2592`，以当前源码和测试为准。对应论文是 [MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs](https://arxiv.org/abs/2512.22219)，不是早期 Mirage 的 Multi-Level Superoptimizer 论文。第一遍的目标是能够解释“一段张量计算如何变成任务图，并在 GPU 上被正确调度执行”。

建议用 3～4 周、每次 60～90 分钟完成。阶段 0～6 是主线，约 24～32 小时；阶段 7～8 按兴趣选读。时间只是学习预算，不是截止日期。每次按“概念 15 分钟 → 跟一条代码路径 40 分钟 → 画图或回答问题 20 分钟”的节奏进行。

阅读中区分两件事：纸面跟踪源码不需要 GPU；运行测试需要兼容的 GPU、CUDA 与已安装的 Mirage。这里列出的测试已静态阅读，未在此次导读整理中实际运行，也未启动构建或下载模型。

完整导读入口：

| 阶段 | 具体材料 | 学习产物 |
|---|---|---|
| 0 | [基础概念与源码对照](stage-0-guide-zh.md) | 执行模型与概念图 |
| 1 | [RMSNorm 最小端到端测试](stage-1-guide-zh.md) | 数据、构图、编译、执行流程 |
| 2 | [Python 到生成的 CUDA](stage-2-guide-zh.md) | 注册链、生成链与地址偏移表 |
| 3 | [任务图与事件生成](stage-3-guide-zh.md) | GCD/LCM 例子与预派发前后对照 |
| 4 | [GPU 运行时与同步](stage-4-guide-zh.md) | worker、scheduler 和迭代时序图 |
| 5 | [RMSNorm 的 CUDA 实现](stage-5-guide-zh.md) | 拷贝组、共享内存与归约图 |
| 6 | [MLP 到完整解码循环](stage-6-guide-zh.md) | 模型 shape 与请求状态图 |
| 7 | [多 GPU 通信与远程依赖](stage-7-guide-zh.md) | 数据和完成信号的跨卡时序图 |
| 8 | [Profiling 与性能证据](stage-8-guide-zh.md) | 测量范围图与实验记录模板 |

本地版本有两个运行前需核对的具体限制：默认拆分 worker/scheduler 的启动路径与 NVSHMEM 路径之间存在配置不一致；内置 profiler 在拆分 grid 下共享记录缓冲区，按当前索引公式存在冲突风险。阶段 7、8 分别给出源码依据与静态学习路线，不把这些路径当成已运行验证通过，也没有为导读修改核心实现。

**先建立一张全局地图**

```text
模型 / 小测试描述计算
  ↓
Python PersistentKernel：张量、算子、切块映射
  ↓ Cython 接口
C++：依赖分析、任务划分、事件生成、CUDA 代码生成
  ↓
task_graph JSON + CUDA 源码 → 编译后的运行模块
  ↓
GPU scheduler：在每轮开始时预先把任务放进 worker 队列
  ↓
GPU worker：取任务、等待必要依赖、调用 CUDA device 函数
  ↓
完成计数 → 解锁已派发任务的依赖等待
控制事件 → 下一轮推理 / 终止
```

MPK 的 task 可以先理解成“一块张量上的计算工作”，worker 是反复执行 task 的常驻执行者。task 不等于一个独立 CUDA kernel launch，也不等于一块永远固定的物理 SM。论文的 SM 级视角，要落到源码里的 CTA、队列和任务描述符上理解。

以下路径均相对于仓库根目录。第一遍暂时跳过 `src/search/`、大部分 `src/transpiler/`、MoE、FP8、推测解码和复杂 GEMM 模板；这些不是理解 MPK 调度主线的前置条件。

**阶段 0：补齐恰好够用的概念，2～3 小时**

具体学习材料：[阶段 0 导读：概念讲解、源码对照、画图练习与自测](stage-0-guide-zh.md)。

阅读入口：`README.md` 的 About、How MPK Works；`include/mirage/persistent_kernel/runtime_header.h` 的 `TaskDesc`、`EventDesc`、`RuntimeConfig`，只看字段名称和注释。

需要补齐的概念：

- CUDA：grid、CTA/thread block、warp、SM 的区别；block 内同步与 block 间同步；global/shared memory；kernel launch。
- 并发：生产者/消费者、队列、原子计数器；先写数据再发布“可读”的含义。acquire/release 的细节留到阶段 4。
- 图：节点、边、前驱/后继、拓扑顺序、fork（分支）与 join（汇合）。
- LLM：prefill 处理提示词，decode 逐步生成；KV cache 保存历史 K/V；一层里的 RMSNorm、Attention、MLP、残差。
- Python：只需读懂函数、类、字典、tensor shape；Cython 暂时视为 Python 调用 C++ 的桥。

练习：画两张 `A → B → C` 的执行图：一张是三个普通 kernel，另一张是常驻 worker 从队列中依次取任务。标出谁发起下一步、数据放在哪里、依赖由谁保证。

过关标准：能说清 MPK 希望减少 launch 和算子边界带来的开销，并在依赖允许时交错执行不同算子的任务；“融合进常驻 kernel”并不意味着中间张量全部放进 shared memory，也不保证任何形状都会加速。

**阶段 1：读懂最小端到端实例，2～3 小时**

具体学习材料：[阶段 1 导读：RMSNorm 测试逐段讲解、编译产物与自测](stage-1-guide-zh.md)。

按顺序阅读：

1. `tests/runtime_python/test_mode/test_rmsnorm_testmode.py`
2. `python/mirage/mpk/persistent_kernel.py`：`get_default_init_parameters()`、`attach_input()`、`rmsnorm_layer()`、`compile()`、`__call__()`。
3. 回到测试，理解 PyTorch reference 和结果比较。

这份测试没有大模型权重，主线很清楚：创建随机张量 → 构造 `PersistentKernel` → 注册张量 → 添加 RMSNorm → 编译 → 执行 → 与参考结果比较 → 释放资源。

重点回答：

- `attach_input()` 把哪些信息交给图？它是否已经执行计算？为什么输出张量也通过这个入口注册？
- `grid_dim=(batch_size, 1, 1)` 如何描述这层的计算任务？它与常驻 runtime 的 worker 数有什么区别？
- `test_mode=True` 对运行轮次有什么影响？
- `compile()` 与 `pk()` 分别做什么？

环境就绪后，可单独运行：

```bash
python tests/runtime_python/test_mode/test_rmsnorm_testmode.py
```

先原样运行，不要随意缩小 hidden dimension：算子对 shape、线程数和对齐可能有约束。测试通过 `compile(output_dir=...)` 保存 `test_rank0.cu`、`task_graph_rank0.json` 等产物；当前测试默认保存到测试文件所在目录。学习副本可改成独立输出目录，避免后续测试覆盖同名文件。

练习：逐行标记哪些语句属于“准备数据”“构图”“编译”“执行”；记录输入/输出的 shape、dtype 和所属内存。没有运行环境时，先完成这份静态跟踪。

过关标准：能不看实现，复述上述端到端流程。注意 README 中的简化构造示例与当前参数定义已有差异，配置应参考本地测试及默认参数函数。

**阶段 2：打通 Python → C++ → 生成代码，3～4 小时**

具体学习材料：[阶段 2 导读](stage-2-guide-zh.md)。

沿两条短路径阅读，不要通读整个大文件。

构图路径：

```text
PersistentKernel.rmsnorm_layer
  → TBGraph 的输入/输出切块描述
  → kn_graph.customized / register_task
  → src/kernel/graph.cc 中对应任务注册分支
  → TaskRegister::register_rmsnorm_hopper_task（target_cc >= 90 分支）
```

编译路径：

```text
PersistentKernel.compile
  → python/mirage/_cython/core.pyx：generate_task_graph
  → src/kernel/runtime.cc：Graph::generate_task_graph
  → register_mugraph / print_task_graph
  → JSON、CUDA 源码、nvcc 编译与模块加载
```

重点阅读 `src/kernel/task_register.cc` 中选中的 RMSNorm 注册函数，以及 `src/kernel/runtime.cc` 中生成 `_execute_task` 的片段。

重点回答：

- `KNGraph`、`TBGraph` 分别描述哪一层的信息？
- `input_map` 中 `(0, -1, -1)` 如何把 grid 轴映射到 tensor 维度？`-1` 表示该 grid 轴不切分该输入，而不是“负索引”。
- 为什么这条手工注册任务的路径里，输出也会被描述为 `tb_graph.new_input(...)`？继续看注册函数怎样按参数位置区分 input/output，不能只凭函数名判断。
- `task_type` 选择哪类工作，`variant_id` 区分什么具体实现？
- 哪些 shape 被写进 CUDA 模板参数，哪些指针由任务描述符提供？

练习：从生成的 CUDA 中找到 `rms_norm_hopper_impl` 与 `_execute_task`，反向定位到生成它们的 C++。若无生成产物，先阅读 `CodeKeeper` 写入字符串的语句，手工拼出对应调用。

过关标准：能把一个 Python 层调用追到最终 CUDA device 函数，并区分“生成 CUDA 的 C++”与“在 GPU 上执行的 CUDA”。这条入口通过模型 builder 和任务注册选择实现；不要把它理解为任意 PyTorch 程序都会自动转换。

**阶段 3：理解依赖如何变成任务与事件，4～5 小时**

具体学习材料：[阶段 3 导读](stage-3-guide-zh.md)。

按顺序阅读：

1. `include/mirage/persistent_kernel/runtime_header.h`：`FullTaskDesc`、`TaskDesc`、`EventDesc`。
2. `include/mirage/kernel/annotated_graph.h`：数据结构和 `build_annotated_graph()` 前的 pipeline 注释。
3. `src/kernel/annotated_graph.cc`：先看 most-recent-writer、边构造、拓扑顺序，再看切块映射。
4. `src/kernel/runtime.cc`：`register_mugraph()`、`dfs_create_events_add_tasks()`、`print_task_graph()`。
5. `tests/runtime_python/test_mode/test_diamond_fork_join_testmode.py`；需要时配合 fork、join、negative 测试。

第一遍只处理一条直链，理解张量读写关系如何变成任务依赖；第二遍处理 `A → {B, C} → D`。GCD/LCM 的推导放在第二遍，先知道它们在协调不同算子的切块边界。

重点回答：

- 同一 tensor 多次被写入时，消费者依赖哪个写入者？为什么只有 tensor 名字还不够？
- 一个算子怎样展开成多个 task？一个 event 为什么可能等待多个 producer task？
- `num_triggers`、`first_task_id`、`last_task_id` 如何表达后继任务的启动条件与范围？
- `trigger_event` 与 `dependent_event` 各自做什么？后者的等待检查在哪里执行？
- 残差边的“去除”何时只是删除冗余调度依赖，为什么并不等于删除残差数值计算？

练习：手工建立“两项生产任务共同完成后，才能执行一项消费任务”的事件表，逐次更新计数器。再对照 diamond 测试绘图，记录真实 task 数与 event 数。

注意：diamond 测试中的某些输入采用完整复制映射，因此可能需要整个生产算子完成后才能推进；这份测试不能单独证明细粒度跨算子重叠。当前分析器还会拒绝部分 fork/join 角色组合，阅读 `test_case2_case3_negative_testmode.py` 了解边界，不要假设任意 DAG 都支持。

过关标准：给出两个算子的切块方式，能解释哪些消费者要等待哪些生产者，以及为什么不会过早启动。可选地搜索 `MIRAGE_DUMP_ANNOTATED_GRAPH`，按其实现导出分析后的图。

**阶段 4：精读 GPU runtime，这是主线重点，5～6 小时**

具体学习材料：[阶段 4 导读](stage-4-guide-zh.md)。

主文件：`include/mirage/persistent_kernel/persistent_kernel.cuh`。推荐跳转顺序：

1. `persistent_kernel()`：先看 CTA 如何被分配到 worker / scheduler 角色。
2. `execute_worker()`：取 task、获取描述符、检查依赖、`_execute_task()`、触发完成事件。
3. `execute_scheduler()`：取 event、选择任务范围、分配到 worker 队列。
4. `compute_task_id()`、`get_task_iteration_num()`：理解任务索引与迭代轮次。
5. `prepare_kernel()`、`terminate_schedulers()`：理解如何启动与收尾。
6. `launch_persistent_kernel()`：最后补齐 host 端 launch 路径。

配套文件：`runtime_header.h` 中的队列/计数器字段；原子与内存序辅助函数按实际引用跳转。

先写下简化伪代码，再回源码补例外：

当前编译器会在 `register_mugraph()` 末尾预派发普通任务，并把许多中间事件改成 `EVENT_EMPTY`，由 worker 等依赖计数。因此下述 scheduler 派发主要由开始等控制事件驱动，不能理解成每个算子完成后都重新派发下一个算子。

```text
worker:
  从队列取 task
  必要时等 dependent_event 达到当前轮次的计数
  执行对应算子
  更新 trigger_event 的计数
  满足触发条件时发布 event

scheduler:
  从队列取 event
  将对应范围的 task 分配给 worker
  或处理本轮结束、下一轮、终止
```

重点回答：

- worker 和 scheduler 传递的是 task/event 标识，还是整块 tensor 数据？
- “已经入队”为什么不总等于“依赖已经全部满足”？
- 为什么写入队列内容与更新 ready 指针的顺序不能随意交换？
- 结果写入 → release 发布 → acquire 观察 → 读取结果，这条顺序在哪里体现？为什么 `__syncthreads()` 不能替代跨 CTA 通信？
- 为什么事件计数与 iteration 有关？如何避免上一轮完成状态被当成下一轮完成状态？
- 增加 scheduler 数量为什么不一定更快？常驻角色会消耗哪些资源？

练习：画一张时序图，参与者只有 producer worker、scheduler、consumer worker。标出写数据、计数更新、event 入队、task 入队、等待依赖和实际读取数据的位置。

过关标准：能完整解释一项 task 从入队到执行、再到解锁后继的生命周期，并找到保证正确性的同步语句。

版本细节：源码既有一个 `persistent_kernel` 容纳两种角色的路径，也有分开的 `worker_kernel` / `scheduler_kernel` 路径，当前初始化默认设置 `split_worker_scheduler=true`；启动前还有准备 kernel。因此论文的 megakernel 概念不能机械理解为程序从初始化到结束只出现一次 CUDA launch。第一遍先借单 persistent kernel 路径理解角色，再对照实际拆分路径。

**阶段 5：深入一个 CUDA 任务，3～4 小时**

具体学习材料：[阶段 5 导读](stage-5-guide-zh.md)。

继续使用 RMSNorm，避免同时学习调度和复杂矩阵乘法。

阅读入口：

- `src/kernel/task_register.cc`：当前架构选中的 RMSNorm 注册函数。
- `include/mirage/persistent_kernel/tasks/hopper/rmsnorm_hopper.cuh`：`rms_norm_hopper_impl()`。
- Ampere 路径对应 `include/mirage/persistent_kernel/tasks/ampere/rmsnorm.cuh`；以 Python 分支和生成代码为准。

先从公式理解计算：`y = x * rsqrt(mean(x²) + eps) * weight`。再依次看数据搬运、每线程局部求和、warp/block 归约、归一化和写回。第一次遇到异步拷贝时，把目标收窄为“搬了哪些数据、何时能读取、哪里等待”，暂不展开所有 PTX 细节。

重点回答：

- 输入指针是否已经指向当前任务的 tile？偏移由生成代码还是 device 函数计算？
- 每个线程处理哪些元素，如何覆盖尾部元素？
- 为什么这里是 `__device__` 函数，而不为每项 task 再 launch 一个 `__global__` kernel？
- shared memory 在一次任务内怎样使用，任务之间的中间结果存在哪里？
- 测试 reference 与 CUDA 的 eps、累加精度、输出 dtype 是否一致？

练习：对一行输入画出线程分工与归约过程。记录当前 Hopper 注册调用使用的 eps 与测试 reference 默认 eps 的差异；理解现有容差通过不等于逐位一致。

过关标准：能解释“一项 task 内部如何算”和“多项 task 之间如何调度”这两个层次。之后若想读 GEMM，再选择一种架构补 CUTLASS/CuTe、TMA、Tensor Core；`docs/mpk/tma.md` 可作为后续材料。

**阶段 6：把机制放回 LLM 推理，5～7 小时**

具体学习材料：[阶段 6 导读](stage-6-guide-zh.md)。

按顺序阅读：

1. `tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py`：先读 `test_gateup_only()`，再读 `test_gateup_silu_down()`。
2. `python/mirage/mpk/models/qwen3/builder.py`：`new_intermediate_tensors()`、`build_layers()`。
3. `python/mirage/mpk/mpk.py`：`MPK.build()`、`generate_task_graph()`、`compile()`、`__call__()`。
4. `demo/qwen3/demo_mpk_wrapper.py`：看配置、构图、编译、加载请求和调用之间的关系。
5. 回到 `persistent_kernel.cuh`：offline 对应的 `prepare_next_batch()`、scheduler 对 `EVENT_END_OF_TASK_GRAPH` 的处理。

先固定单 GPU、dense 模型、普通解码的阅读分支。MLP 先用随机权重测试，不需要下载 Qwen3。完整 demo 的执行留到硬件、依赖和模型容量确认之后。

重点回答：

- gate/up 两个线性变换如何组织，SiLU 与乘法处理什么形状，down projection 与 residual 在哪里相遇？
- 每层 Attention 的 Q/K/V、KV cache、输出投影分别对应哪些图节点？先追 shape 和数据流，再读 attention 内部算法。
- `MPK` 的模型包装层与 `PersistentKernel` 的底层构图/执行层分别负责什么？
- 权重、中间激活、KV cache、step、tokens、分页元数据，哪些在构图时确定，哪些在运行时变化？
- 本轮图结束后，谁更新请求状态、生成下一轮输入、决定继续还是退出？

练习：画出一层 Transformer 的数据流，并给每个节点标一个 builder 调用；再画一轮 decode 结束到下一轮开始的控制流。分别解释 `test_mode` 单轮执行与实际 serving 循环。

过关标准：能从“输入 token”出发，讲清任务图如何执行一轮模型计算，并回到“继续生成或终止”。第一遍到这里，已经完成 MPK 核心实现的闭环。

**阶段 7：选读多 GPU 通信，3～4 小时**

具体学习材料：[阶段 7 导读](stage-7-guide-zh.md)。

阅读入口：`python/mirage/mpk/multigpu.py`；Qwen3 builder 的 `world_size > 1` 分支；`PersistentKernel.allreduce_layer()`；`task_register.cc` 中实际选中的 NVSHMEM 注册函数；`tasks/hopper/allreduce.cuh` 或 `tasks/blackwell/allreduce.cuh`；runtime 中远程事件与远程 scheduler 分支。

先补 tensor parallel 的列/行切分，以及 all-gather / all-reduce 的作用，再跟踪一条“局部计算 → 通信 → 后继计算”的链。定位远程数据写入、完成通知、消费者等待，说明通信何时与独立计算重叠。不要只因为它们在同一大 kernel 中就断言存在重叠。

练习：为两张 GPU 画数据与通知分别跨卡的时序图；注明哪些缓冲区需要 NVSHMEM 的远程访问能力。没有多卡环境也可以完成源码分析。

过关标准：能解释多卡正确性依赖哪些额外机制，并区分“本地任务完成”与“远程数据已可供消费”。

**阶段 8：用 profiling 验证理解，2～3 小时**

具体学习材料：[阶段 8 导读](stage-8-guide-zh.md)。

阅读入口：`include/mirage/persistent_kernel/profiler.h`、runtime 的 `PROFILER_EVENT_START/END`、`python/mirage/mpk/profiler_persistent.py` 的 `export_to_perfetto_trace()`；参考 demo 的 `--profiling` 参数。

练习：在环境允许时选择已验证正确的小工作负载，观察 worker 空洞、任务持续时间和依赖边界。一次只改一个受支持的变量，例如某层任务划分或 worker/scheduler 配比；先检查正确性，再比较时间线。不可随意突破架构或配置约束。

必须区分：编译耗时、预热耗时、GPU 执行耗时、端到端推理耗时。当前 offline 的 `prepare_next_batch()` 对 profiling/test mode 有特殊终止处理，解释 trace 时要核对实际执行轮数；插桩时间线也不能直接代替无插桩性能数据。

过关标准：提出一个具体观察，例如“这里等待上游结果”，并用源码依赖与时间线共同解释；不能只凭一张图断言性能瓶颈。

**每阶段保留一页笔记**

- 这段代码要解决什么问题？
- 输入、输出、状态分别是什么？
- 在 CPU 还是 GPU 上执行？发生在构图、编译还是运行时？
- 谁调用它，它把控制权交给谁？
- 正确性依赖哪个条件、计数器或同步动作？
- 一个具体的小例子能否走通？

找符号时优先使用 `rg -n '符号名' 路径`。本文以符号为导航，避免代码更新后行号漂移。注释中的文档路径也要先确认存在；例如当前 `annotated_graph.h` 引用了本地未提供的 `parallel_path_design.md`，此时直接对照实现与 negative tests。

若后续需要安装或重编译，先检查逻辑 CPU、可用内存、系统负载和临时盘空间，再设置适度并发。起步通常不超过 `min(16, 逻辑CPU数/4)` 个有效编译 worker，模板较重时进一步降低，并计入嵌套 NVCC 并发；前几分钟监测 CPU、进程总内存与临时盘增长。阅读阶段无需为了开始学习而立刻构建整个仓库。

第一次学习直接从阶段 0 的双执行图和阶段 1 的 RMSNorm 测试开始。后续逐阶段讨论时，每次只追一条调用链，先预测行为，再用源码验证。
