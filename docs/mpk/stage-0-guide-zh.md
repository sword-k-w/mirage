**MPK 阶段 0 具体导读：从基础 CUDA 走到任务图与常驻执行**

这份导读是 [分阶段阅读计划](reading-guide-zh.md) 的阶段 0，基于本地提交 `97de2592`。你只需要基础 C++ / CUDA 知识；不需要安装新依赖、编译项目或准备模型权重。文中的小矩阵和伪代码用于推演，不是可以直接运行的 MPK 程序。

学完后，你应当能解释这句话里的每个名词：

> MPK 把张量计算组织成有依赖的任务，由 GPU 上常驻的 worker 执行，通过事件和 scheduler 推动后续计算。

先不用证明调度算法，也不用读懂 Tensor Core 算子。遇到不了解的模板、宏和位运算，只要不影响本节的问题，就标记后继续。

建议准备一页笔记，按以下顺序读，合计约 155 分钟，可以拆成两次：

| 步骤 | 时间 | 完成后留下什么 |
|---|---:|---|
| 1．从普通 CUDA 看 MPK 要解决的问题 | 15 分钟 | 普通执行与常驻执行两张图 |
| 2．把 CUDA 名词对齐 | 25 分钟 | thread / warp / block / SM / task 对照表 |
| 3．把算子图细化成任务图 | 25 分钟 | 一张 9 个任务的依赖图 |
| 4．理解队列、计数器与同步 | 25 分钟 | 一个事件的状态变化表 |
| 5．认出源码里的三个结构体 | 20 分钟 | 关键字段的中文注释 |
| 6．补齐 LLM 与 Python 的最小背景 | 25 分钟 | decode 流程图与 shape 笔记 |
| 7．完成自测 | 20 分钟 | 独立回答问题，再核对答案 |

**1．先从你熟悉的三个 kernel 开始**

打开 [README.md](../../README.md)，只读 About 和 How MPK Works，先不跟安装命令。抓住 compiler、runtime、persistent kernel 三个词。论文把编译器与 GPU 内的并行运行时作为共同工作的两部分：前者生成任务图和代码，后者执行任务。[MPK 论文摘要](https://arxiv.org/abs/2512.22219)

假设有一个形状为 `[3, 4]` 的矩阵 `X`，计算分成三步：

```text
A：Y = X + 1
B：Z = Y * Y       // 逐元素平方
C：O = 2 * Z
```

传统 CUDA 实现可以写成下面的形式。为突出执行关系，省略了内存分配、检查和具体参数：

```cpp
A<<<grid, block, 0, stream>>>(X, Y);
B<<<grid, block, 0, stream>>>(Y, Z);
C<<<grid, block, 0, stream>>>(Z, O);
```

对于这里的普通同一 stream 执行方式，设备按顺序执行 A、B、C。CPU 通常可以连续提交这些操作，不需要每次 launch 后都调用 `cudaDeviceSynchronize()`；kernel launch 通常相对 host 异步。因此不要把开销想成“CPU 每次都等 GPU 算完才提交下一项”。[CUDA 异步执行说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

```text
CPU：提交 A ─ 提交 B ─ 提交 C ─ ……可以继续做别的事
GPU：执行整个 A ─ 执行整个 B ─ 执行整个 C
```

这里有两个值得思考的问题：每一步都要提交一个 kernel；B 即使只需要 A 的一部分结果，在这条普通执行路径上也要经过整个 A 的完成边界。

MPK 把工作拆成 task，让长期运行的 GPU 执行代码反复取任务。下面是概念伪代码，不代表真实线程分工：

```text
CPU：准备任务图，启动 GPU runtime

GPU worker：
  循环：
    取一项 task
    等待它需要的依赖
    执行对应的 device 函数
    报告完成
    收到终止任务时退出

GPU scheduler：
  根据完成事件，把后续 task 分配给 worker
```

这是入门用的事件驱动概念图。当前源码会进一步在每轮开始时预派发普通任务，并让 worker 等待依赖计数；中间完成事件未必重新经过 scheduler。阶段 3、4 会对照这个具体实现展开。

“persistent”指执行者在多个任务之间保持运行；不是它永不退出，也不是权重会永久留在寄存器里。“mega-kernel”描述把许多计算/通信任务纳入大的执行结构，并不要求把所有计算手工展开成一个无法分辨算子的函数。

实际仓库还有准备 kernel，以及分开启动 worker / scheduler 的可选路径。本阶段先按单 persistent kernel 理解。

还有一个边界：这个逐元素例子本来就能直接融合为简单 CUDA 表达式，MPK 并非这种小例子的必要方案。选它只是因为依赖容易画清楚；真正需要理解的是如何组织包含不同算子、多个 tile 和复杂依赖的大计算。现有系统也有 CUDA Graphs 和其他融合手段，不能把普通 CUDA 一概想成未经优化的逐算子提交。

停下来写一句话：**“在 MPK 中，CPU 启动之后，谁决定后续任务何时执行？”**

**2．把 CUDA 的执行单位与 MPK 的任务分开**

先读这张表。前五项属于 CUDA 的执行与硬件概念；后三项属于当前 MPK 的软件设计。

| 名词 | 本阶段的理解 | 读 MPK 时不要混淆什么 |
|---|---|---|
| thread | 一个 CUDA 线程 | 一个 task 通常由多个线程合作执行 |
| warp | 同一个 block 内的 32 个线程组成的执行组 | 不等于一个独立 kernel |
| block / CTA | 协作线程组，可以使用 block 级 shared memory 和同步 | 是软件执行单位，不是硬件 SM 的别名 |
| grid | 一次 kernel 启动的所有 blocks | grid 中的 blocks 不一定同时驻留 |
| SM | GPU 上承载线程执行的硬件单元 | `blockIdx.x` 是 block 编号，不是物理 SM 编号 |
| task | 由 MPK 描述的一份计算或通信工作 | 不会因为新增一项 task 就必然新增一次 kernel launch |
| worker | 反复领取和执行 task 的执行者 | 在这里的主要执行路径上由一个 CTA 承担，可以顺序执行多项 task |
| scheduler | 处理事件并给 worker 派发任务的 GPU 代码 | 与 GPU 硬件里的 warp 调度器是不同层次的东西 |

普通 CUDA 中，一个 block 的线程在一个 SM 上执行；一个 SM 可以同时承载多个 block，实际数量受寄存器、shared memory 等资源限制。warp、block、grid 是组织线程的层次，SM 是承载它们的硬件。[CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

做一个算术练习：`kernel<<<3, 128>>>()` 有 3 个 block，每个 block 4 个 warp，总计 384 个线程。仅凭这个表达式，不能断言它使用了 3 个不同的物理 SM，也不能断言 3 个 block 同时开始。

再看 MPK：假设逻辑上有 100 项计算任务，运行时配置 8 个 worker。这意味着少量常驻 worker 要分多轮处理这些任务；不意味着必须 launch 100 个 worker CTA。

在图构建 API 中看到的某层 `grid_dim`，主要描述该层怎样划分工作；启动整个 persistent runtime 时的 grid 则承载 worker / scheduler。**它们描述不同层次，不能拿某层的 `grid_dim.x` 直接当成 runtime 的 worker 数。** 实际任务展开还有通信子任务等例外，阶段 0 先用“一份 tile 工作对应一项 task”理解普通计算路径。

现在打开 [persistent_kernel.cuh](../../include/mirage/persistent_kernel/persistent_kernel.cuh)，搜索 `void persistent_kernel(`。只看下面这段实际分支：

```cpp
if (blockIdx.x < config.num_workers) {
  execute_worker(config);
} else {
  execute_scheduler(config, -(SCHEDULERS_PER_BLOCK * config.num_workers));
}
```

你现在只需要读出：同一次启动中的 CTA 被划分成两类角色。第二个参数的编号转换先跳过。`SCHEDULERS_PER_BLOCK = 4` 是当前代码的配置，表示 scheduler CTA 可以容纳多个 scheduler；并非“每个 scheduler 都是一个完整 CTA”。

接着分清存储位置：

| 存储 | 本阶段关心的作用 | 一个重要限制 |
|---|---|---|
| register | 保存线程计算中的局部值 | 不能当作另一个 worker 可以直接访问的公共数组 |
| shared memory | 支持同一 CTA 中线程协作与数据复用 | 普通 block 级 shared memory 不是整个 grid 的共享存储 |
| global memory | 保存权重、张量、队列与跨 worker 的状态 | 共享地址并不自动保证读写顺序正确 |

这里使用普通 block 的模型；cluster 等扩展机制留到后续需要时再读。MPK 可以让不同 worker 接力处理同一个张量，因此中间数据仍可能写到 global memory。“在一个 kernel 中执行”不会自动消除这些读写。

这一小节的练习：画出 `SM ← worker CTA ← 多个线程`，再在 worker 旁边画出 `task 0 → task 1 → task 2`。前者表示硬件承载与线程组织，后者表示一个执行者按时间处理多项工作。

**3．从三个算子，走到九个任务**

继续用 `[3, 4]` 的矩阵，把每一行作为一个 tile，即一块工作数据：

```text
X[0, :] → A0 → Y[0, :] → B0 → Z[0, :] → C0 → O[0, :]
X[1, :] → A1 → Y[1, :] → B1 → Z[1, :] → C1 → O[1, :]
X[2, :] → A2 → Y[2, :] → B2 → Z[2, :] → C2 → O[2, :]
```

算子图只有 `A → B → C` 三个节点。任务图有九个节点，每个节点注明“算哪一步、算哪一行”。图中的箭头表示数据依赖：只有前面产生的数据可用，后面的计算才可以读取。

现在问自己：**B0 是否一定需要等待 A1 和 A2？**

这个例子里不需要，因为 B0 只读取 Y 的第 0 行。若两个 worker 的计算时长都简化为每项一个时间格，可以有如下合法顺序。它只展示可行性，不预测 MPK 的具体派发次序，也忽略调度开销：

| 时间格 | worker 0 | worker 1 |
|---|---|---|
| 0 | A0 | A1 |
| 1 | B0 | A2 |
| 2 | C0 | B1 |
| 3 | B2 | C1 |
| 4 | C2 | 空闲 |

时间格 1 同时出现 A 和 B 的任务。这就是跨算子交错执行的一种直观形式：下游所需的局部数据已经完成时，可以与上游其他工作重叠。

但是，如果把 B 改成“每项任务都需要完整 Y”，就必须等待 A0、A1、A2 全部完成。能否流水执行由真实读写依赖和切块方式决定，不能仅凭“使用 MPK”得出结论。

再补三个读图术语：

- DAG：有方向、没有环的图。本轮张量计算可以用它表达。
- 拓扑顺序：任何节点都排在它的前驱之后；它是一种合法顺序，不等于实际 GPU 上唯一的时间顺序。
- fork / join：fork 是一个结果被多个后继使用；join 是一项计算需要多个前驱的结果。

```text
      ┌→ B ─┐
A ────┤     ├→ D
      └→ C ─┘
```

图中 D 要等 B 和 C。这里的多前驱等待不是选择“先完成的那个”。一个推理程序可以反复执行本轮 DAG；迭代控制不需要被误画成本轮数据依赖图中的非法环。

练习：抄下九任务图，把每个节点需要读取和写入的张量区域标在旁边；然后把 B0 改成读取 Y 的全部三行，重新画它的依赖边。阶段 3 会进一步解释编译器怎样从切块映射生成这些关系。

**4．理解 GPU 上“报告完成”是什么意思**

先把 worker、scheduler 和数据分开画：

```text
                      任务编号
GPU scheduler ───→ worker 队列 ───→ GPU worker
      ↑                                  │
      └──────── scheduler 事件队列 ←──────┘ 完成通知

张量数据：worker 通过 task 中的指针读写相应内存
```

队列主要传递 task/event 标识，不是把整个 activation tensor 一遍遍搬进队列。具体计算任务通常通过描述符中的指针访问数据。

假设 B 必须等 A0、A1 都完成，我们用一个教学事件 E 表达这件事。只考虑第一轮，初始计数为 0，每个生产任务完成后贡献 1 次触发：

| 时刻 | 已发生的事 | E 的计数 | 是否满足两次触发 |
|---|---|---:|---|
| 开始 | 都没有完成 | 0 | 否 |
| 下一步 | A1 完成 | 1 | 否 |
| 再下一步 | A0 完成 | 2 | 是，可以推动 B |

完成顺序不重要，两个结果都可用才重要。这就是后面 `EventDesc.num_triggers` 所描述的门槛。真实 runtime 还考虑事件类型、迭代轮次和远程通信；不能把这个表直接扩展成“每轮都把所有计数器清零”。

这里需要三个同步概念。

**第一，原子更新避免丢失计数。** 两个 worker 若都执行普通的“读 counter、加 1、写回”，可能都读到 0，最终都写回 1。原子加确保更新不会这样互相覆盖。原子加解决的是这个共享数值的并发更新问题。

**第二，数据必须先可用，再发布可消费的状态。** 假设生产者要写 Y，再发布完成通知。消费者看到完成后就可能读取 Y，因而需要建立“结果写入 → 完成状态发布 → 消费者观察 → 结果读取”的顺序。普通 `bool ready` 或 `volatile` 本身不能替代完整的同步协议；一个原子操作也不会无条件保证所有其他内存都已按你期望的方式可见。要看 memory order、作用域及配套的 block 同步。[CUDA C++ Memory Model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)

本阶段把 release 理解成“按规定发布之前的写入”，把 acquire 理解成“按规定观察发布并约束后续读取”即可。真正的正确性需要配对的同步关系，阶段 4 再追完整链条。

可以在 `persistent_kernel.cuh` 中搜索两个实际符号，认出它们的作用，不必展开 PTX：

```text
atom_add_release_gpu_u64  ← 本地完成事件计数的原子更新
ld_acquire_sys_u64        ← 某些依赖等待路径对计数的读取
```

**第三，block 内同步不等于跨 block 同步。** `__syncthreads()` 协调同一 block 的线程；worker A 的 block 调用它，并不能让 worker B 的 block 也一起到达同一个屏障。MPK 要额外建立跨执行者的依赖协议。

同理，不能随便把很多普通 blocks 改成“所有 block 都在 while 中等其他 block”。若等待者占满可驻留资源，而它等待的生产者还没被调度，就可能无法继续。这也是常驻运行时需要认真规划 worker/scheduler 资源与执行进度的原因。

最后留意一个源码事实：任务入队后，worker 仍可能检查 `dependent_event` 并等待。因此，“拿到任务”与“可以执行任务”是两个动作。阶段 0 不必理解所有提前派发规则，但不要把两个状态合并成一个。

练习：画四步时间线“写结果 → 发布完成 → 观察完成 → 读结果”，再写下：哪里需要原子性，哪里需要可见性，哪些线程需要先在 block 内同步。此时不用自己实现无锁队列。

**5．第一次读三个核心结构体**

打开 [runtime_header.h](../../include/mirage/persistent_kernel/runtime_header.h)。可直接搜索下面三个定义，暂时跳过宏、构造函数、`union` 和 TMA 字段：

```bash
rg -n 'struct.*TaskDesc|struct EventDesc|struct RuntimeConfig' include/mirage/persistent_kernel/runtime_header.h
```

先读 `TaskDesc`：它描述“要做哪项工作，以及从哪里读写数据”。

| 字段 | 先这样理解 | 给自己提的问题 |
|---|---|---|
| `task_type` | 任务种类，例如某种归一化或矩阵乘法 | worker 该调用哪类代码？ |
| `variant_id` | 同一种类里选择的实现版本 | shape 等不同配置如何选择具体实现？ |
| `input_ptrs` / `output_ptrs` | 指向输入/输出数据 | 描述符装的是数据本身，还是地址？ |
| `dependent_event` | 若有效，执行前需要等的事件 | 我需要什么结果先准备好？ |
| `trigger_event` | 完成后要贡献触发的事件 | 我的完成会推进哪项依赖状态？ |

一个 task 可以参与包含多个依赖的图，并不意味着这里有一个任意长度的依赖数组。编译器怎样把关系组织进这些字段，以及能支持哪些组合，是后续阶段的问题。

再读 `EventDesc`：它描述“累计到什么条件后，进行什么动作”。

| 字段 | 含义 |
|---|---|
| `event_type` | 事件满足条件后处理哪类动作，如派发任务或处理图结束 |
| `num_triggers` | 一轮所需的触发次数 |
| `first_task_id` / `last_task_id` | 对相关派发事件，表示一段后继任务范围；当前循环使用左闭右开范围 |

**MPK 的 `EventDesc` 不是 CUDA API 的 `cudaEvent_t`。** 它是项目自己定义的任务依赖/调度描述。`EventDesc` 也没有保存不断增长的当前计数；当前计数另存于 runtime 的数组中。

最后读 `RuntimeConfig`，先只看这几组：

```text
num_workers、num_local_schedulers        执行角色的数量
all_tasks、all_events                    描述符数组
all_event_counters                      事件当前计数
all_event_num_triggers                  事件所需触发次数
worker_queues、sched_queues              任务队列与事件队列
first_tasks                             启动任务的信息
step、tokens、input_tokens、output_tokens 推理状态和输入输出
```

可以把它理解成 runtime 访问共享状态的一组参数和指针。`RuntimeConfig config` 按值传给函数，并不意味着它指向的整个任务图和所有张量都被复制了一遍。

练习：不要照抄整份结构体，只把上述字段抄到笔记，用自己的话解释。能区分“描述符”“编号”“指针所指的数据”“可变计数器”，就完成本节目标。

**6．把这些概念接到 LLM 与 Python 入口上**

先理解一个简化的 decoder-only Transformer。设 `B` 是 batch 中的序列数，`T` 是本次处理的序列长度，`H` 是隐藏维度。规则批次下可把激活看成 `[B, T, H]`；实际 serving 经常把本次要处理的 token 打包成 `[本次 token 总数, H]`，再配合索引元数据。

```text
token ID
  → embedding：变成向量
  → 多层 Transformer
  → 最终归一化、词表投影
  → logits：词表中每个候选 token 的分数
  → argmax 或 sampling
  → 下一个 token ID
```

以常见的 pre-norm dense 层为例，可先看下面的概念表达式；它省略了位置编码、模型特定归一化等细节：

```text
h = x + Attention(RMSNorm(x))
y = h + MLP(RMSNorm(h))
```

你需要记住各部件的职责：RMSNorm 按特征做归一化；Attention 让当前 token 使用上下文信息；MLP 进行逐 token 的非线性特征变换；残差把原输入加回来。阶段 0 不需要推导 attention，也不需要复现 FlashAttention。

用提示词 `[p0, p1, p2]` 理解 prefill / decode：

```text
prefill：处理 p0、p1、p2，建立各层 KV cache
         末位置的输出可用于选出第一个新 token q0

decode 第一步：输入 q0，使用已有 cache，选出 q1
decode 第二步：输入 q1，继续使用并扩展 cache，选出 q2
……
```

KV cache 保存各层历史 token 的 key/value，减少重复计算这些历史 K/V。它不表示 Attention 不再读取历史数据，也不会自动让每一步成本与上下文长度无关。普通单 token decode 每条活跃序列本次只输入一个新 token；多请求批处理、分块 prefill、推测解码会让实际批次更复杂，留到阶段 6。

这与 MPK 的联系是：重复推理轮次通常复用一套计算结构，但 token、位置、缓存和请求状态在变化。任务描述中的计算结构、runtime 中的可变推理状态，需要分开理解。GPU 常驻执行减少部分组织开销，也要付出调度、同步和常驻资源的成本；是否更快仍取决于工作负载。

接着打开 [RMSNorm 最小测试](../../tests/runtime_python/test_mode/test_rmsnorm_testmode.py)。只读到 `pk.rmsnorm_layer(...)`，借此补齐 Python 语法：

```python
params = PersistentKernel.get_default_init_parameters()
params["test_mode"] = True
pk = PersistentKernel(**params)
```

`params` 是“名字 → 值”的字典，`**params` 把字典展开为命名参数。`pk` 是类实例；`pk.rmsnorm_layer(...)` 是方法调用，定义里的 `self` 指这个实例。`pk()` 则会调用它的 `__call__()` 方法。

再看张量：

```python
x = torch.randn(batch_size, hidden_dim, dtype=dtype, device="cuda")
x_dt = pk.attach_input(x, name="x")
```

`x` 是实际张量；在当前源码中，`attach_input()` 获取 shape、stride、dtype 并关联底层张量，返回构图所用的 `DTensor` 描述。构图对象与数值存储有关联，但不是同一个概念。`torch.randn` 等数据准备操作本身可以在 GPU 上执行；这里说的“构图尚未执行算子”，专指添加 MPK 图节点不等于已经执行相应模型计算。

将本次测试的输入 shape `[16, 4096]` 读成 16 行，每行 4096 个特征。`dtype` 决定元素表示方式；对普通连续行主序张量，stride 可以是 `(4096, 1)`，表示移动一行/一列分别跳过多少个元素。无需先学完整 PyTorch。

Cython 暂时只记住：它是 Python 调用底层 C++ 图实现的桥。看到 `.pyx` 文件时，不必立刻学习如何编写扩展模块。

练习：将“模型权重、任务描述、KV cache、当前 token、事件计数”分成两组：本次运行中主要复用的信息，以及随着任务/推理轮次变化的状态。注明这个分类是为了理解当前路径，不是所有模式下都绝对不变。

**7．合上源码，自测后再看答案**

先独立回答，优先用图和短句：

1. 普通同一 stream 上连续提交 A、B、C，CPU 必须在每次提交后等待吗？B 在设备端是否因此就可以任意早于 A 执行？
2. 100 个 task、8 个 worker，是否要启动 100 个 worker CTA？一项 task 是否就是一个 warp？
3. 三行逐元素例子中，A0 完成、A1/A2 未完成，B0 可以开始吗？若 B0 改成读取全部三行呢？
4. A0、A1 同时报告完成，为什么普通 `counter++` 可能出问题？把它换成原子加，是否已经证明整个张量读写协议正确？
5. 为什么 producer block 调用 `__syncthreads()`，不能代替 consumer block 的依赖等待？
6. 一个 task 已进入 worker 队列，能否断言它现在就可执行？
7. `TaskDesc` 存放完整输入张量吗？`EventDesc` 存放实时计数吗？MPK event 是否就是 `cudaEvent_t`？
8. 融合到一个 persistent kernel 后，中间激活是否都在 shared memory 中？
9. prefill 之后，第一个新 token 从哪里来？接下来的 decode 以什么作为新输入？KV cache 保存什么？
10. `attach_input()`、`rmsnorm_layer()`、`compile()`、`pk()` 分别属于什么阶段？

综合题：把 `A → B → C` 画成两张图。一张使用三个普通同 stream kernel，一张使用常驻 worker 和 scheduler。每张都必须标出：谁提交/派发工作，数据在哪里，谁保证依赖，何时允许 B 的第一项任务执行。无需画寄存器布局或推导 PTX。

下面是核对答案：

| 题目 | 应包含的要点 |
|---|---|
| 1 | CPU 通常可连续异步提交；普通同 stream 的设备执行仍有顺序约束 |
| 2 | 8 个 worker 反复处理任务；task 是逻辑工作，不等于 warp |
| 3 | 按行独立时可以；读取全矩阵时要等所需三行全部可用 |
| 4 | 非原子读改写可能丢更新；还需证明结果可见性、发布顺序及相关同步 |
| 5 | `__syncthreads()` 的协作范围是同一 block，不能形成任意跨 block 屏障 |
| 6 | 不可以，worker 可能仍要等待 `dependent_event` |
| 7 | task 中主要是类型、变体、事件和指针等描述；当前计数另存；两种 event 不同 |
| 8 | 不一定，跨 worker 的中间结果仍可能经由 global memory 传递 |
| 9 | 普通生成流程用 prefill 末位置输出选择首个新 token；下轮输入该 token；cache 存历史各层 K/V |
| 10 | 关联数据与构图描述、添加计算节点、生成/编译运行代码、执行已构建的计算 |

完成综合图，并能解释第 2～8 题，就具备进入阶段 1 的概念基础。若卡在 task 与 worker 的区别，回读第 2 节；若卡在“何时可以开始”，回读第 3、4 节。暂时不理解 `variant_id` 怎样生成、事件 ID 怎样编码，不影响进入下一阶段。

下一次从 RMSNorm 测试开始，沿“创建张量 → 注册张量 → 添加一层 → compile → 调用”的路径逐行读。把这一页概念笔记放在旁边，遇到一个结构体字段，就尝试在自己画的图上指出它的位置。

继续阅读：[阶段 1 具体导读：读通 RMSNorm 最小端到端测试](stage-1-guide-zh.md)。
