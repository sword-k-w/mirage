**MPK 阶段 3 导读 从张量读写关系到任务和事件**

承接 [阶段 2](stage-2-guide-zh.md)，目标是解释“为什么某项 task 必须等待这些生产者，编译器怎样把等待写进描述符”。基于 `97de2592`，建议 4～5 小时，先直链、再分支汇合、最后看预派发转换。

这一阶段主要在 CPU 端。GPU 怎样轮询、执行和发布结果，留到 [阶段 4](stage-4-guide-zh.md)。以下小整数例子是手算模型，不是本次运行产生的任务图。

**1．建立三个层次的对照**

| 层次 | 表达什么 | 源码入口 |
|---|---|---|
| 算子与 tensor 关系 | 谁读取了谁写入的张量 | [annotated_graph.cc](../../src/kernel/annotated_graph.cc) 的 `build_annotated_graph()` |
| 分块关系 | 哪些 producer tile 对哪些 consumer tile 提供数据 | [annotated_graph.h](../../include/mirage/kernel/annotated_graph.h) 的 `EdgeInfo`、`TaskView` |
| 执行描述 | task 的输入输出、事件、触发计数 | [runtime.cc](../../src/kernel/runtime.cc) 的 `register_mugraph()` |

先看 `annotated_graph.h` 末尾的 pipeline 注释，随后按照源码中的 `Step (a)` 到 `Step (j)` 跳读。注释引用的 `parallel_path_design.md` 在本地未提供；以实现和测试为依据。

**2．为什么要寻找最近一次写入**

一个 tensor buffer 可以被多层复用。如果只以 GUID 作为唯一生产者标识，后写入可能与先读取的关系混淆。

手算以下顺序，`buf` 表示同一块被复用的存储：

```text
A：写 buf
B：读 buf → 写 tmp
C：读 tmp → 再写 buf
D：读 buf → 写 out
```

正确关系是 `A→B→C→D`。B 读取时最近的 writer 是 A，D 读取时最近的 writer 是 C。把所有 `buf` 的读取都绑定到 C，可能制造错误依赖甚至假环。

在 Step (a) 中找 most-recent-writer；理解 `EdgeInfo` 为什么用 producer layer/output slot 与 consumer layer/input slot 描述边，而不能只用 tensor GUID。再读 Step (b) 的拓扑环检查，知道它是防御性验证。

这个算法不是任意内存程序的完整别名证明。如果程序绕过图接口复用未知指针，不能仅靠最近写入规则保证正确。阶段 1/2 的张量关联、view 描述和参数角色都必须成立。

练习：把上例的 D 移到 C 前面，重新标出每次读对应的 writer。先按注册顺序解释，再画图。

**3．残差边为什么可以从调度图中去掉**

考虑 `A→B→C`，同时 C 还读取 A 的输出，形成一条 `A→C` 边。如果 C 已经必须等 B，而 B 又必须等 A，那么完成顺序的约束可以通过路径传递。

Step (c) 对这类存在替代路径的边做 residual stripping。这里删去的是冗余调度约束；C 的输入参数、读取操作和残差加法仍然保留。不要把它解释为数学上删掉 residual。

阅读 [test_residual_stripping_testmode.py](../../tests/runtime_python/test_mode/test_residual_stripping_testmode.py)，记录原图与剥离后图的变化。进一步的 tile 等待由后续映射生成，因此正确性必须把完整读写图和分块关系一起看，不能凭一个“有路径”的口号随意删边。

**4．先手算 GCD 为什么会出现**

假设桥接 tensor 沿某一维长 12，producer 均匀分成 4 个 tile，每个覆盖 3 个元素；consumer 分成 6 个 tile，每个覆盖 2 个元素。

```text
producer：P0 [0,3) P1 [3,6) P2 [6,9) P3 [9,12)
consumer：C0 [0,2) C1 [2,4) C2 [4,6)
          C3 [6,8) C4 [8,10) C5 [10,12)
```

源码按共享切分边界分组：`event_dim = gcd(4,6) = 2`。两组分别覆盖 `[0,6)` 与 `[6,12)`。

| 事件组 | producer 任务 | consumer 任务 | 本轮所需触发数 |
|---|---|---|---:|
| E0 | P0、P1 | C0、C1、C2 | 2 |
| E1 | P2、P3 | C3、C4、C5 | 2 |

这是一个安全的分组策略，不是对每个元素求出的最细依赖：C0 实际只需 P0 的区域，但在这个事件分组中也随整组等 P1。由此理解“细粒度”是相对于整层屏障而言，并不保证粒度绝对最小。

在 Step (g) 中核对：

- `event_dim` 按 tensor 维度记录事件分区数。
- `last3` 按 grid 的 x/y/z 轴记录每个事件覆盖多少任务。
- `input_map/output_map` 连接这两个坐标系。

本例 producer 侧每组 2 个任务，consumer 侧每组 3 个。若一侧沿该 tensor 维度完全不分块，partition 为 1，GCD 会退化为 1，得到更粗的等待关系。涉及 view 的 `is_barrier_edge` 分支也会将事件维度降为 1。

练习：改成 producer 4 份、consumer 4 份；再改成 consumer 完整读取。分别算事件组数量和每组触发次数。答案依次是 4 个单 producer 组，以及 1 个等待 4 个 producer 的组。

**5．fork 和 join 为什么还需要 LCM**

fork 是同一 producer 的结果被多个不同 consumer 使用；join 是同一 consumer 需要多个不同 producer。角色分类在 residual stripping 之后进行，不能直接拿原图的边数判断。

同一 fork producer 的不同分支可能要求不同的事件分组。若 producer 有 12 项任务，分支 B 每组需要 2 项 producer，分支 C 每组需要 3 项 producer，为统一组边界，LCM 得到每组 6 项。这样组更粗，但能满足两个分支的共同约束。

这对应 Step (h) 对 producer-side `last3` 的统一；Step (i) 在 join 的 consumer 一侧做对称处理。LCM 作用在每组任务跨度，不要误说成“对 tensor 元素值做 LCM”。

阅读 [diamond 测试](../../tests/runtime_python/test_mode/test_diamond_fork_join_testmode.py)：

```text
A RMSNorm
  ├→ B Linear ─┐
  └→ C Linear ─┴→ D LinearWithResidual
```

先在纸上标出 A 的 fork 和 D 的 join，再检查 B/C 的输入映射。该测试某些线性层完整读取激活，因此事件会退化成整组等待；它验证分支汇合正确性，不能单独证明细粒度计算重叠。

当前编译器会拒绝特定角色组合。配合 [negative 测试](../../tests/runtime_python/test_mode/test_case2_case3_negative_testmode.py) 阅读 Step (e)：case 2 是 join-consumer 同时为 fork-consumer；case 3 是 join-producer 同时为 fork-producer。把它们记成当前实现的检查边界，不要据此推导所有 DAG 都可表示。

头文件部分注释对 `trigger_event/dependent_event` 的文字说明容易混淆，错误信息中也带有特定分析阶段的命名。最终字段含义以 runtime 的读写为准：worker 在执行前等 `dependent_event`，完成后触发 `trigger_event`。不要根据一条注释反转这两者。

**6．任务与事件怎样写进容器**

回到 `register_mugraph()`，先看 `build_tasks_bid_lex` 与各层 task map，再找 `dfs_create_events_add_tasks()`。后者递归遍历事件分区，确定 producer/consumer 的 bid 子范围，形成 task 与 event 的关联。

在最简单的边上，先按下面的中间表示理解：producer 的 `trigger_event=E`；E 记录 `num_triggers` 与 consumer 的连续 task 范围 `[first_task_id,last_task_id)`。fork 分支为了让每组 consumer 能落在连续范围，会交错组织任务；因此不能假设同一算子的任务在所有图中都占一段连续数组。

图的所有叶节点都必须参与 `EVENT_END_OF_TASK_GRAPH` 的完成计数。对于两个没有汇合的输出分支，只等最后插入的分支会过早宣布整图完成。源码在这里遍历每个 leaf 的任务；配合 [fork 测试](../../tests/runtime_python/test_mode/test_fork_point_testmode.py) 理解。

**7．一定读到最后的 Prelaunch 转换**

这是当前实现与“每完成一项就派发下一项”概念模型的重要差别。`register_mugraph()` 末尾的 `Prelaunch all tasks` 会：

1. 把开始事件的任务范围设成所有普通任务，从索引 2 到末尾。
2. 将中间的 `EVENT_LAUNCH_TASKS / EVENT_LAUNCH_MASSIVE_TASKS` 改成 `EVENT_EMPTY`。
3. 为这些事件覆盖的 consumer 填入 `dependent_event`。

因此最终执行主要是：**每轮开始时预先派发任务，worker 取到任务后再等数据依赖；producer 完成仍更新计数，但 `EVENT_EMPTY` 不再请求 scheduler 派发一遍消费者。** 开始、结束、终止等控制事件仍有调度作用。

用 E0 的手算例子对照：

| 对象 | 转换前 | 转换后 |
|---|---|---|
| P0/P1 | 完成后触发 E0 | 仍触发 E0 |
| E0 | 满足次数后派发 C0～C2 | 成为 `EVENT_EMPTY`，保留完成计数用途 |
| C0～C2 | 由 E0 解锁派发 | 可预先入队，但必须等 E0 达到本轮门槛 |

不要在 `EVENT_EMPTY` 上停下来认为“这项依赖被删掉了”。它从派发条件变成了 worker 的执行前等待条件。阶段 4 要用这个最终表示解释运行时。

**8．观察输出和做自测**

可搜索 `MIRAGE_DUMP_ANNOTATED_GRAPH`。当前实现设置为非 `0` 时，将分析图的文本摘要写到 stderr；虽然头文件有 JSON 的描述，实际函数返回的是人可读文本，不应将它当成 `task_graph.json`。

若环境已就绪，在已有测试或学习副本运行前设置：

```bash
MIRAGE_DUMP_ANNOTATED_GRAPH=1 python tests/runtime_python/test_mode/test_diamond_fork_join_testmode.py
```

这会触发测试编译和运行，沿用阶段 1 的资源检查。无需为了阅读把整个测试目录并行跑一遍。没有 GPU 时，按上述数值例子手算即可。

| 自测题 | 核对要点 |
|---|---|
| buffer 复用为什么不能只看 GUID？ | 不同读取应绑定当时最近的写入者 |
| residual stripping 删了什么？ | 冗余调度边，未删除数学输入与残差运算 |
| 4 份生产、6 份消费得到几组？ | 在均匀同维切分例子里为 2 组；每组 2 个 producer、3 个 consumer |
| GCD 分组一定是最细依赖吗？ | 不是，可能额外等待组内数据 |
| LCM 统一的是什么？ | fork/join 对应侧每事件覆盖的任务跨度 |
| 为什么入队不等于可执行？ | prelaunch 后还要等 `dependent_event` |
| `EVENT_EMPTY` 是否不再计数？ | 仍由生产者更新，只是不派发新任务 |
| 图结束等谁？ | 所有终端分支的叶任务 |

过关笔记：最近 writer 示例、GCD 表、fork/join 图、prelaunch 前后对照各一份。接下来阅读 [阶段 4](stage-4-guide-zh.md)。
