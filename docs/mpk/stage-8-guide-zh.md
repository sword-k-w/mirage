**MPK 阶段 8 导读 用时间线验证执行与性能理解**

这是 [总计划](reading-guide-zh.md) 的最后一阶段，可在读完单卡主线后直接学习。目标是从源码确认测量范围，再分析任务时间线；不以“产出一张漂亮图”作为结论。基于 `97de2592`，建议 2～3 小时。

本地默认拆分 launch 与内置 profiler 存在需要处理的兼容性问题，见第 4 节。因此本阶段提供完整的静态阅读与 CPU 分析练习，GPU 采集部分明确列出前提。本次没有采集 trace、执行 benchmark 或验证硬件性能。

**1．先给你的问题选一种时间**

| 问题 | 应观察的量 | 不应直接替代它的量 |
|---|---|---|
| 首次使用要等多久？ | 环境准备、代码生成、编译、加载等分阶段耗时 | 单个 task duration |
| 一项算子工作执行了多久？ | 已知起止点的任务区间 | 完整请求 wall time |
| 一轮模型计算多久？ | 一轮图的起止与同步边界 | 所有 worker task 时间的简单相加 |
| 用户得到 token 的延迟？ | 明确定义的端到端请求或逐 token 时间 | 只含 GPU 数学部分的 trace |

并行任务的时间之和可以大于实际经过时间。解读任何“快了多少”之前，写明起点、终点、是否编译、是否预热、批量/序列长度、执行轮数、是否开启插桩。

**2．从插桩位置读测量口径**

打开 [persistent_kernel.cuh](../../include/mirage/persistent_kernel/persistent_kernel.cuh)，搜索 `PROFILER_EVENT_START` 和 `PROFILER_EVENT_END`，先读 worker 的普通任务区间：

```text
取任务与描述符
  → 等 dependent_event
  → CTA 同步
  → task START
  → _execute_task 或 BEGIN 的空操作
  → CTA 同步
  → task END
  → 更新完成计数 / 必要时发布 scheduler 事件
```

因此 task 条带不是“从入队到完成”的全部延迟。它包含分派后的任务执行和区间内同步，但不包含前面的队列等待、描述符准备、依赖等待；后面的完成事件发布也在另一个位置。

scheduler 的 prepare-batch 等事件、worker 的调度事件记录各有自己的插桩。每个条带必须回到源码找起止点，不能根据名字将所有空白都解释成 memory stall 或调度器低效。

本例有些 `TASK_BEGIN_TASK_GRAPH` 也会被记录，即使没有数学工作。stage 2 的 task type、variant、实际 tensor tile 与 trace 的显示标签不是完全相同的信息，后面还要确认标识对应关系。

练习：把阶段 4 的任务生命周期画成七段，给被普通 task START/END 包围的部分涂色。未覆盖的地方写“未直接测得”，不要填入猜测时间。

**3．读 profiler 的编码和导出**

主文件：[profiler.h](../../include/mirage/persistent_kernel/profiler.h)、[profiler_persistent.py](../../python/mirage/mpk/profiler_persistent.py)。

先看 `ProfilerEntry`：64-bit 记录拆成两个 32-bit 字段。第 0 条记录保存 block/group 数；其他记录保存 tag 和低 32 位 global timer。字段名 `delta_time` 容易误导，这里写入的是 timer 值，并非已算好的 duration。

tag 的字段按实际位移读：

| 位段 | 内容 |
|---|---|
| 高 13 位 | event_no，同一记录者的事件序号 |
| 接下来的 8 位 | block/group 编号 |
| 接下来的 9 位 | 事件/任务类型编号 |
| 最低 2 位 | BEGIN、END、INSTANT |

`PROFILER_INIT` 让各 block/group 使用交错的槽位写入：起点为 `1+block*num_groups+group`，每次前进 `num_blocks*num_groups`。这解释了导出器为什么先读 header，再扫描数组中的非零项。数组中的原始排列不等于全局时间排序。

`_decode_events()` 将 GPU buffer 拷到 CPU 并解码；`export_to_perfetto_trace()` 生成 block/group tracks；`export_to_csv()` 配对 BEGIN/END，输出：

```text
task_type_id, task_type_name, block_idx, group_idx, event_no,
begin_ts, end_ts, duration_ns
```

CSV 导出会报告未配对 BEGIN/END；Perfetto 导出按 event name 表查名字。新 task type 若没有同步到名称表，可能出现不同导出器处理不同的情况，应核对枚举与映射，而不是删掉不认识的记录。

时间取自低 32 位计数，约每 4.3 秒回绕。CSV 通过模 `2^32` 计算配对区间 duration，但整条时间线的原始排序仍需考虑回绕；不能将长 trace 的绝对时间当成单调无限增长。event_no 与 block/group 字段也有编码上限。

**4．当前版本采集前必须检查的具体问题**

先把源码事实与推断分开：

| 源码事实 | 对测量的含义 |
|---|---|
| 初始化设置 `split_worker_scheduler=true` | 默认是两个独立 grid |
| 两个执行函数都使用 `config.profiler_buffer` | 指向同一个记录缓冲区 |
| `PROFILER_INIT` 按各自 grid 的 blockIdx/gridDim 计算 header、起点与步长 | 两个 grid 都从 block 0 开始，索引空间并未区分 |
| scheduler 只有 warp 0 写记录 | 其他 scheduler warp 的活动不直接出现在该 block 的记录中 |
| 宏中没有按已分配容量停止写入的检查 | 预估容量不足可能写越界，而不只是少几个条带 |

由前三项可作出一个明确的静态推断：**默认 split 路径下，worker 与 scheduler 的 profiler 写地址存在重叠，header 也可能被不同 grid 的值覆盖。** 例如各自 block 0 的首个记录都指向 buffer 的第 1 项。这不是本次实验观测到的 trace，而是依据地址公式发现的风险。

因此不要只在原测试中添加 `profiler_tensor` 就直接相信输出。真正采集前，应先确认采用了已验证的独立记录区方案，或者已验证的单 grid 配置；改变 launch 方式本身会影响资源与性能，需要记录，不能把另一配置的结果当成默认路径结果。本次任务是导读，没有修改 runtime 来修复这个问题。

此外，offline 的 `prepare_next_batch()` 对 `MPK_ENABLE_PROFILING` 使用了特殊请求完成分支，可能让请求在一轮后结束。开启 profiler 不只是增加几条 store，还可能改变工作量。对比无插桩运行时，要确保轮数和有效 token 数一致。

练习：假设 worker grid=96 blocks，scheduler grid=12 blocks、num_groups=1，分别列 block 0 的前三个写槽：worker 为 `1,97,193`，scheduler 为 `1,13,25`。第一个槽已经冲突；后续也不能共享一份 header 解释两种步长。

**5．满足采集前提后再接入小测试**

使用阶段 1 的已验证 RMSNorm 学习副本，先保留原数据、shape 与正确性检查。完成上一节配置核验后，采集流程是：

1. 计算实际记录者数量和每个记录者的条目预算，分配清零的 `torch.uint64` CUDA buffer。
2. 在创建 `PersistentKernel` 前给 `params["profiler_tensor"]` 和 `trace_name` 赋值。
3. 新建对象并编译，核对 `-DMPK_ENABLE_PROFILING`；不能给已编译模块临时换个 Python 布尔值就假定设备端有插桩。
4. 运行同一小图并确认 GPU 工作完成；检查数值结果。
5. 检查 BEGIN/END 完整性、记录数、类型和 block/group 范围，再打开 trace。

单一正确布局下的容量可按 `1 + num_blocks*num_groups*entries_per_group` 估算，单位是 64-bit 元素；每段 BEGIN/END 至少占两条，还要算进控制与调度事件。留余量但不要用一个固定的“每个 GPU 都适用”的常量。

当前 `PersistentKernel.__call__()` 会在设置 profiler 时调用导出函数。内含 CPU 拷贝、编码与文件写入，因此用 host wall time 包围 `pk()` 会把导出成本一起算进去。计时必须区分“运行 + 导出”与“仅 GPU 工作”。

重复采集还需要确认清零和重新初始化方式；不要让上一轮残留非零记录混入下一份 trace。编译仍遵守阶段 1 的 CPU/内存/负载/临时盘检查，不并行启动大量变体。

**6．不用 GPU 也能完成的时间线练习**

下面是人为构造的时间区间，单位任意，不是 MPK 测量结果：

| worker | 任务 | 开始 | 结束 |
|---|---|---:|---:|
| 0 | A0 | 0 | 4 |
| 0 | B0 | 5 | 9 |
| 1 | A1 | 0 | 7 |
| 1 | B1 | 8 | 12 |

手算三件事：

- 四段执行时间总和是 `4+4+7+4=19`。
- 观察窗口跨度是 `12-0=12`，不是 19。
- 若只将这些非重叠区间当成两个 worker 的“已记录执行”，覆盖比例是 `19/(2×12)≈79.2%`。

这个覆盖比例不是 GPU SM occupancy、Tensor Core 利用率或显存带宽利用率；窗口还忽略了其他未插桩活动。worker 0 的 `[4,5)` 空白可能有依赖等待、描述符准备或其他控制开销，单凭图不能确定是哪一种。

再画合法依赖 `A0→B0`、`A1→B1`：每条边都满足 producer 先完成。若改成 B0 必须等待 A0 和 A1，B0 在时间 5 开始就与依赖矛盾；这时应检查图映射、时钟/记录解码或执行正确性，不能把矛盾解释为“优化得更快”。

此练习说明时间线与任务图应相互校验。但实际 trace 的 event_no 是记录计数，不保证等于全局 task index；必须额外建立任务映射，不能仅凭同样的 type 名称猜出它属于哪个 tile。

**7．分析一份已验证的 CSV**

下面代码只读现有 CSV，用于概览计数、总时长和类型分布；路径替换为真实已验证的输出文件。它不需要 CUDA，也不重跑工作负载：

```python
import csv
from collections import defaultdict
from pathlib import Path

path = Path("stage8_verified.csv")
groups = defaultdict(list)
with path.open(newline="") as f:
    for row in csv.DictReader(f):
        groups[row["task_type_name"]].append(int(row["duration_ns"]))

for name, durations in sorted(groups.items()):
    durations.sort()
    print(name, "count=", len(durations),
          "sum_ns=", sum(durations),
          "min_ns=", durations[0],
          "max_ns=", durations[-1])
```

先比较相同类型的离散程度，再回到 block/group 时间线找最慢条带。即使某类型累计时间最多，也不能马上宣称它决定端到端延迟：关键路径、并行度和依赖等待同样影响总时间。

性能解释按下面的证据梯度写：

| 观察 | 可以写的结论 | 尚需补的证据 |
|---|---|---|
| 某 worker 有长空白 | 该区间没有记录到普通 task 执行 | 更细的取任务/等待插桩或外部工具，才能解释原因 |
| 某类 task 时间离散大 | 同类记录的时长不一致 | shape、有效 token、tile 边界、竞争与同步差异 |
| 不同算子的条带重叠 | 在可信时钟与映射下存在执行区间重叠 | 数据依赖是否允许，以及这种重叠对总时间的收益 |
| task 总时长下降 | 被记录工作的累计时间下降 | 同工作量的无插桩端到端对比 |

只有硬件计数器等额外证据才能支持具体 pipe/带宽/occupancy 判断。内置 timeline 不直接给这些指标。

**8．只设计一次可解释的比较**

选一项已满足算子约束的变量，例如阶段 1 的行数变化，或已经理解并验证过的任务划分。一次只改一个变量，记录正确性和实际工作量。改变行数时工作量也变了，不能把延迟差简单说成同任务的优化收益。

可复用下面的实验记录表：

| 字段 | 应填内容 |
|---|---|
| 代码与硬件 | commit、GPU、CUDA/依赖版本、实际 launch 路径 |
| 输入 | shape、dtype、请求/token 数、上下文长度、随机种子 |
| 配置 | worker/scheduler、任务 grid、测试/服务模式、实际轮数 |
| 正确性 | reference、eps、误差口径与结果 |
| 测量 | 计时起止、插桩与否、预热和重复次数 |
| 观察 | 数值与 trace 中直接可见的事实 |
| 解释 | 与源码对应的理由，以及仍未证实的假设 |

不要把编译时间计入 steady-state GPU 执行时间，也不要拿 profiling 的单轮与正常 serving 的多轮做百分比比较。硬件状态与其他进程干扰也需要记录；无需为了导读立即启动性能优化任务。

**9．最终自测**

| 自测题 | 核对答案 |
|---|---|
| 普通 task 区间包含 dependent_event 等待吗？ | 当前 worker 插桩在等待之后开始，不包含 |
| CSV task 时间总和等于一轮 latency 吗？ | 并行区间会重叠，不能直接等同 |
| `block_idx` 能直接当物理 SM 编号吗？ | 不能，它来自 kernel 的 block 标识 |
| event_no 是完整 TaskId 吗？ | 不是，是记录序号 |
| 32-bit timer 的限制是什么？ | 会回绕，长 trace 需要额外处理；当前导出有明确限制 |
| 默认 split 路径只开 profiling 是否足够？ | 不足，先解决/核验共享 buffer 索引与 header 冲突 |
| 有 CSV 且没有配对错误就证明测量正确吗？ | 不证明，还要检查布局、覆盖范围、映射与工作量 |
| 空白是否等于 GPU 什么都没做？ | 不等，可能是未插桩活动或等待 |
| 从 timeline 能直接读出 Tensor Core 利用率吗？ | 不能，需要相应硬件计数证据 |

完成本阶段的成果是一份测量范围图、一份纸面时间线分析和一个实验记录模板。回到 [总阅读计划](reading-guide-zh.md)，检查你是否已经能从 Python 调用一路解释到 GPU 计算、依赖、推理循环和性能证据。
