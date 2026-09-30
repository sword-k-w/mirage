**MPK 阶段 4 导读 GPU 上的任务执行和调度**

承接 [阶段 3](stage-3-guide-zh.md)。这次把预派发后的任务图放进 GPU runtime，解释一项工作如何取出、等待、执行、报告完成，并走到下一轮或退出。基于 `97de2592`，建议 5～6 小时，分三次读：worker、scheduler、启动与收尾。

主文件是 [persistent_kernel.cuh](../../include/mirage/persistent_kernel/persistent_kernel.cuh)，配套 [runtime_header.h](../../include/mirage/persistent_kernel/runtime_header.h) 与 [mpk_atoms.cuh](../../include/mirage/persistent_kernel/mpk_atoms.cuh)。第一遍固定单 GPU、offline、无推测解码，远程通信留到阶段 7。

**1．先画角色和状态**

```text
all_tasks / all_events：静态描述
all_event_counters：随运行增加的完成计数

scheduler → worker_queue → worker
    ↑                         │
    └──── sched_queue ← 控制事件发布

worker 执行前还会直接读取 dependent_event 的计数
```

当前 prelaunch 策略下，中间普通依赖往往由计数器完成，不必绕一圈 scheduler。图开始、图结束、终止等仍使用事件队列。

先读 `persistent_kernel()` 的角色分支，再读 `worker_kernel()` 与 `scheduler_kernel()`。前者在一次 grid 中划分两类 CTA；后两者分别启动。当前初始化写死 `split_worker_scheduler=true`，实际默认走拆分路径。两种方式调用的核心执行函数仍是 `execute_worker()` 和 `execute_scheduler()`。

在 scheduler 中找 `threadIdx.x % 32 == 0`：当前每个 scheduler 由一个 warp 的 lane 0 执行控制循环，每个 scheduler CTA 安排 `SCHEDULERS_PER_BLOCK=4` 个这样的角色。不要将它想成全 block 的线程都在共同消费同一份控制逻辑。

**2．先读 ID 再读数组下标**

`TaskId` 同时装迭代轮次和任务表索引：

```text
TaskId = (iteration_num << 32) | position_index
高 32 位：第几轮
低 32 位：all_tasks 中哪一项
```

例如同一个普通任务位置 7，在第 1 轮与第 2 轮的 ID 不同，但都查同一份 task 描述。不要拿完整 TaskId 直接当 `all_tasks` 数组下标。跟读 `compute_task_id()`、`get_task_iteration_num()`、`get_task_position_index()`。

EventId 则还编码所属 GPU 和远程标记；单卡先通过 `get_event_position_index()` 取出位置。不需要背所有常量，但要知道 `EVENT_INVALID_ID` 不能作为普通数组索引解引用。

**3．worker 如何拿到一份描述符**

在 `execute_worker()` 中按以下顺序读：

1. `worker_id=blockIdx.x`，选择本 worker 的队列。
2. 维护本地取队列位置以及 shared memory 中的描述符缓存。
3. 读取 `worker_queue_last_ready_task_id`，判断有没有已发布条目。
4. 从环形队列取 TaskId，提取位置，再从 `all_tasks` 搬描述符。
5. 等待异步描述符拷贝完成，并让 CTA 线程同步后使用。

队列名字里的 `last_ready_task_id` 容易误导：这里用于判断可读边界的是队列位置计数，不等同于 payload 中编码后的 TaskId。分别给“第几个槽”和“槽里是哪项任务”起笔记变量名。

`task_descs` 的 shared buffer 可以批量缓存多份描述符，减少每执行一个任务就重新准备全部调度信息的成本。它不是提前算好了这些任务的 tensor 输出。

环形队列位置按 `% per_worker_queue_len` 回绕，但逻辑位置计数继续前进。因此物理槽 0 会被重复使用，不能用槽号是否相同判断任务是否重复。源码有队列容量断言；这不是可以无界增长、任意生产速度下都安全的通用队列。

练习：假设教学队列容量为 4，逻辑位置从 0 到 6，对应物理槽为 `0,1,2,3,0,1,2`。同时写出“逻辑消费位置”和“已发布边界”，解释为什么只有数组取模是不够的。

**4．为什么取到了任务仍要等**

紧接描述符取出后，找到 `task_desc->dependent_event`。若它有效，thread 0 计算当前轮需要的计数：

```text
needed_counts = all_event_num_triggers[event_index] × 当前任务迭代轮次
```

单卡普通路径通过 acquire load 轮询，直到 `actual_counts >= needed_counts`；之后 `__syncthreads()` 让 CTA 的其他线程进入计算。

教学例子：事件每轮需要两项生产任务完成，计数在同一次 launch 的不同图迭代间累加：

| 时刻 | 累计计数 | 第 1 轮消费任务 | 第 2 轮消费任务 |
|---|---:|---|---|
| 初始化 | 0 | 等待 2 | 等待 4 |
| 第 1 轮一个生产者完成 | 1 | 等待 | 等待 |
| 第 1 轮都完成 | 2 | 可执行 | 仍等待 |
| 第 2 轮一个生产者完成 | 3 | 已满足 | 仍等待 |
| 第 2 轮都完成 | 4 | 已满足 | 可执行 |

它解释了为什么需要 iteration，为什么不能只检查“计数不为 0”。下一次 host launch 的 `prepare_kernel()` 会清理计数与队列位置，和同一次运行里推进图迭代是两个不同层次。

**5．计算与完成通知之间的顺序**

worker 遇到 `TASK_TERMINATE` 直接退出，遇到 `TASK_BEGIN_TASK_GRAPH` 不做数学计算，普通任务调用阶段 2 读过的 `_execute_task()`。

普通任务返回后有 CTA 同步，再由 thread 0 更新 `trigger_event`。本地路径调用 `atom_add_release_gpu_u64(counter,1)`，返回的是更新前的值，所以判断最后一个完成者时检查 `count+1` 是否等于本轮门槛。

例如门槛为 2，第二次原子加返回旧值 1，`1+1==2` 成立。第一次返回 0，不会重复发布同一轮完成事件。

随后按事件类型分两种常见情况：

- `EVENT_EMPTY`：完成计数已经更新，不再向 scheduler 入队。这是普通 prelaunch 依赖的重要路径。
- 需要 scheduler 处理的事件：分配事件队列槽，写入事件位置，按顺序发布可读边界。

并非每项任务结束都导致一次新派发。把这个事实写回阶段 0 的概念图：那里“报告完成”既可以解锁 worker 等待，也可以推动控制事件。

**6．仔细读发布队列的三个步骤**

worker 向 scheduler 队列发布事件时，多个 worker 可能竞争同一队列。先原子领取逻辑槽，再写槽内容，最后推进 `last_ready_event_id`。

假设 W0 领到槽 5，W1 领到槽 6，但 W1 先写完。若 W1 直接宣布“槽 6 以前全都可读”，scheduler 可能读到尚未写好的槽 5。源码通过 CAS 循环等待发布边界轮到自己的槽，避免跨过未完成的空洞。

```text
领取槽位 → 写 payload → 连续发布 ready 边界
```

这与“更新了某个原子变量，所以其他数据自然都好了”不是一回事。阅读时为每次共享访问标出四项：地址、访问线程、memory order、作用域。

| 辅助函数 | 本阶段要识别的角色 |
|---|---|
| `st_relaxed_gpu_u64` | 写队列 payload，本身不是完整发布协议 |
| `atom_add_release_gpu_u64` | 原子更新与 release 发布 |
| `atom_cas_release_gpu_u64` | 按预期边界发布，避免跳过队列空洞 |
| `ld_acquire_gpu_u64` | 观察已发布的队列边界 |
| `ld_acquire_sys_u64` | 当前依赖等待分支使用的系统作用域 acquire 读取 |

PTX 的作用域与 memory order 要和完整路径一起理解。`__syncthreads()` 负责 CTA 内协作，不替代跨 CTA 的发布/观察；`__nanosleep()` 减少忙轮询压力，不提供正确性保证。本导读指出同步意图与实际调用，不把它当作已经完成的形式化内存模型证明。

**7．scheduler 如何组织派发**

`execute_scheduler()` 给每个 scheduler 分配负责的 worker 范围，循环读取本地事件队列和公共广播队列。公共队列的每个本地 scheduler 有自己的读取进度；不要把它误读为一个全局抢走即消失的普通工作队列。

先只跟这三个事件分支：

| 事件 | 动作 |
|---|---|
| `EVENT_LAUNCH_DEPENDENT_TASKS` | 推进本 scheduler 的 iteration，将开始事件范围中的任务按 worker 分片派发 |
| `EVENT_END_OF_TASK_GRAPH` | 准备下一批数据；有工作则派发下一轮 begin task，否则通知终止 |
| 终止事件 | 本地 scheduler 向自己负责的 worker 队列放终止任务，然后返回 |

普通 `EVENT_LAUNCH_TASKS` 与 massive 分支也保留在运行时代码中，但阶段 3 已看到当前编译器把许多中间事件改成了 EMPTY。区分“runtime 有此能力”和“这个生成图实际会走到此分支”。

预派发让一个 worker 可能取到暂时不能执行的任务。顺序与分配因此影响进展：若在简单自制调度中把依赖消费者排在它唯一能由同一 worker 执行的生产者之前，会自我阻塞。当前编译器的任务排列与运行时派发策略必须一起阅读，不能把“有 while 等待”当成任意乱序都安全的证明。

**8．完整走一次启动与退出**

阅读 `init_kernel()`、`prepare_kernel()`、`launch_persistent_kernel()`、`terminate_schedulers()`，拼出本例控制流：

```text
初始化 runtime 和请求状态
  → prepare_kernel 清队列/计数
  → 人工将 END_OF_TASK_GRAPH 事件放进 scheduler 0 队列
  → scheduler 调 prepare_next_batch，准备第一批
  → 派发第 1 轮 BEGIN task
  → BEGIN 触发公共开始事件
  → schedulers 预派发该轮普通任务
  → workers 等依赖、计算、累加完成计数
  → 所有叶任务完成，产生 END_OF_TASK_GRAPH
  → 下一批 / 终止事件 / workers 返回
```

启动时人工放入 END 事件是一种复用控制逻辑的引导方式，不代表模型在执行前已经计算了一轮。begin task 的索引为 1、termination task 的索引为 0，可结合任务图生成代码核对。

拆分路径使用 prepare-done event 让 worker/scheduler stream 等待准备完成；offline 路径记录两个 done event，再将等待接回调用方 stream。单 kernel 路径则用一个 grid 并在此处调用设备同步。判断 host 何时能读取结果，必须沿实际 stream 路径看，不能只看 `launch_func` 返回。

**9．三张必须独立完成的图**

1. 两个 producer、一个 consumer，共用一个依赖计数器的时序图。
2. 两个 worker 乱序写完事件队列槽，按序发布边界的时序图。
3. 从首次 prepare 到第二轮 begin，再到终止的状态图。

每张图都标出“谁写”“谁读”“等待哪个条件”。这些练习无需 GPU。若在已有测试中加调试输出，只选一个任务或事件，避免全线程 printf 把时序和性能改变得无法解释。

| 自测题 | 核对答案 |
|---|---|
| TaskId 低位和高位分别是什么？ | task 表位置和迭代号 |
| 描述符预取是不是算子预执行？ | 不是，只准备任务控制信息 |
| 为什么普通依赖完成后可能没有 scheduler 消息？ | EMPTY 事件只更新计数，consumer 已预派发 |
| 原子加返回 1，门槛为 2，为什么触发？ | 返回旧值，更新后的值是 2 |
| 领取槽 6 后能否先发布到 7？ | 不能跨过尚未发布的槽 5 等空洞 |
| `__nanosleep()` 是否建立内存可见性？ | 不建立；它与同步语义不同 |
| 初始 END 事件是否证明计算已结束？ | 它用于启动第一批的控制引导 |
| scheduler 是否就是硬件 warp scheduler？ | 不是，它是 GPU 上执行的软件控制逻辑 |

完成后进入 [阶段 5](stage-5-guide-zh.md)，只打开一个计算 task 的内部。
