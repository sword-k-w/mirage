**MPK 阶段 6 导读 从 MLP 子图到完整解码循环**

承接 [阶段 5](stage-5-guide-zh.md)，把编译、任务、算子和运行时放回一个实际模型。基于 `97de2592`，建议 5～7 小时，分成 MLP、Transformer 层、请求循环三次阅读。先固定单 GPU、dense Qwen3、普通 greedy 解码，不同时展开 MoE 或推测解码。

本阶段可以完全静态完成；随机权重 MLP 是可选运行入口，不要求下载模型。这里没有执行完整 Qwen3，也不据源码阅读宣称任何 GPU 配置已经验证可运行。

**1．先读一个三步 MLP**

打开 [test_qwen3_mlp_testmode.py](../../tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py)，先看 `test_gateup_only()`，再看 `test_gateup_silu_down()`。文件还包含其他分段测试，先按函数导航，不用一次执行所有部分。

将 batch 记成 B，隐藏维度记成 H，MLP 中间维度记成 I，数学计算是：

```text
gate = X @ W_gate.T
up   = X @ W_up.T
mid  = SiLU(gate) * up
Y    = mid @ W_down.T + residual
```

SiLU(z)=z×sigmoid(z)，乘法是逐元素相乘。这份 MLP 测试没有包含前置 RMSNorm，不能把三步子图直接当成整个 Transformer 层。

本例 B=8、H=4096、I=2048：

| 张量 | shape | 含义 |
|---|---|---|
| X、residual | `[8,4096]` | 当前激活和残差输入 |
| W_gate、W_up | `[2048,4096]` | 两个投影的权重 |
| 合并后的线性输出 | `[8,4096]` | 2I 个特征，包含 gate/up |
| SiLU-Mul 输出 | `[8,2048]` | I 个中间特征 |
| W_down | `[4096,2048]` | 回投影权重 |
| Y | `[8,4096]` | 加残差后的输出 |

练习：只根据矩阵维度检查每一次乘法是否成立。把“张量 shape 对得上”与“内存中的 gate/up 排列对得上”分开，后者是下一节重点。

**2．为什么 gate 和 up 要 shuffle**

测试不是简单地把所有 gate 行放前面、所有 up 行放后面。它调用 `pk.shuffle_tensors()`，按 `num_groups=num_tasks_gatedup//2` 将 gate/up 权重分组交错，使每个 SiLU-Mul 任务能就近取得对应的两份特征。

先用一个简化布局理解：

```text
概念拼接：G0 G1 G2 G3 U0 U1 U2 U3
分组交错：G0 U0 G1 U1 G2 U2 G3 U3
```

这里每个 Gi/Ui 表示一组行，不是单个元素；实际每组大小由代码参数决定。本测试 2I=4096，`grid_for_linear()` 选择 64 个 gate/up 任务，shuffle 分成 32 组，每组包含 64 行 gate 与 64 行 up；SiLU-Mul 的 grid.x 为 32。

这使你可以把阶段 2 的 variant 与地址描述连起来：数学公式相同，但 tile 输入布局不同，后继算子的地址解释也不同。不要对打包后的 `mlp_mid` 直接做“前一半是 gate、后一半是 up”的切片，除非已按实际 shuffle 还原布局。

测试 reference 用原始 W_gate/W_up 分别计算，避免依赖这个物理排列。中间量在 reference 中较多使用 float，而 MPK 中间缓冲区是 BF16，最终误差要结合各阶段舍入解释，不能只比较最终 dtype。

练习：用 4 个 gate 块和 4 个 up 块画出一组 SiLU-Mul 任务应该读哪些块，再对照 `silu_mul_layer()` 的 `input_map` 和 grid。

**3．从单算子正确性走到子图正确性**

先看 gate/up-only 测试，因为它将问题限制为一层线性计算；再看 gate/up→SiLU 与完整 down+residual。若完整输出错误，应逐个检查中间数学结果和布局，而不是立刻怀疑 scheduler。

测试中的 `max_num_batched_tokens/max_num_batched_requests` 设置为 batch_size；这与阶段 1 RMSNorm 最小测试沿用 1 的配置不同。线性或其他任务可能通过 runtime metadata 取得 active token 数，所以迁移测试时必须对照每个实现实际使用的参数。

这个子图还能练习阶段 3：gate/up 切输出特征，SiLU 对成对分块做局部处理，而 down projection 对中间维做归约。后者未必能在某一小块 SiLU 完成后立刻完成整项下游工作。应根据 `linear_with_residual_layer()` 的实际映射生成依赖，不能只看箭头就假定三层完全流水。

**4．进入 Qwen3 builder 先只看调用骨架**

打开 [qwen3/builder.py](../../python/mirage/mpk/models/qwen3/builder.py)，按顺序读 `build_from_model()` 或 `build_from_config()`、`build_from_dict()`、`new_intermediate_tensors()`、`build_layers()`。第一次从一个 `for i in range(num_layers)` 迭代着手。

`build_from_model()` 在 CPU/Python 端准备权重、配置和位置编码；`build_from_dict()` 注册输入、建立中间张量、加 embedding、逐层构图，再添加输出 head。不是每 decode 一步都重新从 Hub 加载一次模型。

在非 split-K 的单卡分支，可以先画成：

```text
input_tokens → embedding → x
  → RMSNorm → QKV Linear → paged attention
  → output projection + residual → h
  → RMSNorm → gate/up Linear → SiLU-Mul
  → down projection + residual → 下一层 x
  → 最终 RMSNorm → vocabulary Linear
  → argmax partial → argmax reduce → output_tokens
```

`build_layers()` 对 `target_cc==100` 选择 split-K workaround，其他架构走另一条路径。为了理解这张图，可以先读非 split-K 分支；研究本机实际执行时必须回到 target_cc 的真实选择。不能把某分支的融合方式照搬到另一分支。

还有一个命名陷阱：注释可能写着“rmsnorm_linear”，但实际调用是 `rmsnorm_layer()` 再 `linear_layer()`。以执行的 API 调用为准，不把注释里的融合名字当作已采用该融合实现的证据。

练习：给上述每个节点写一个 builder 中的 API 名称，然后给两条残差边标出被保留的原输入。接着看这些边在阶段 3 的依赖分析中是否会成为冗余调度边。

**5．先理解 attention 的数据再读 attention 算法**

本阶段不深入 online softmax、Tensor Core 或 TMA，先回答每个输入是什么：

| 参数 | 作用 |
|---|---|
| 投影后的 Q/K/V | 本次 token 的 query/key/value 信息 |
| q_norm/k_norm | 模型的 Q/K 归一化权重 |
| cos/sin 位置编码 | 位置相关旋转所用的数据 |
| k_cache/v_cache | 各层历史 K/V 的存储 |
| qo_indptr | 本轮各请求在打包 token 中的范围 |
| paged_kv_indptr/indices/last_page_len | 各请求映射到哪些 KV 页及最后一页有效长度 |

builder 中 cache 的概念形状为 `[layer, page, page_size, local_kv_heads, head_dim]`。按层传入 `self.k_cache[i]` 后，任务只处理这一层对应的缓存。

手算打包例子：本轮有两个请求，分别处理 3 个和 1 个 token，则有效 token 总数为 4，`qo_indptr=[0,3,4]`。第一个请求读 input_tokens 的 `[0,3)`，第二个读 `[3,4)`；这与 request ID 在整个请求池中的值不一定相同。

分页再做一个独立例子：page_size=4，一个请求拥有 6 个有效位置，需要 2 页，最后一页有效长度是 2。页索引可能是 `[7,2]`，并不要求物理页连续。只从逻辑 token 位置计算连续地址是不够的。

练习：写出“请求编号 → 本轮 slot → token 范围 → KV 页范围”的四级关系。不要把这些索引都叫 batch index。

**6．分清包装层与底层类**

打开 [mpk.py](../../python/mirage/mpk/mpk.py) 的 `MPK.build()`、`generate_task_graph()`、`compile()`、`__call__()`，再读 [demo_mpk_wrapper.py](../../demo/qwen3/demo_mpk_wrapper.py)。

| 层 | 负责什么 |
|---|---|
| demo / MPKMetadata | 提供模型、请求、内存上限和运行选项 |
| MPK | 选择模型 builder，管理构图、加载请求与编译调用 |
| Qwen3Builder | 把模型权重和结构转换为一系列图 API 调用 |
| PersistentKernel | 通用构图、任务代码生成、初始化与 launch |
| GPU runtime | 执行任务和推进请求/图迭代 |

上层 `MPK.__call__()` 在尚未编译时可以调用自身 `compile()`；阶段 1 的底层 `PersistentKernel.__call__()` 直接使用已准备的 launcher。两者不是同一个类，读调用行为时必须写清对象类型。

第一遍不要运行完整 demo：它包含模型加载与显存需求。阅读时先记录 `build → compile → load_new_request → 调用`，再沿请求数据去 runtime，足以理解控制路径。

**7．一轮图结束之后发生什么**

回到 [persistent_kernel.cuh](../../include/mirage/persistent_kernel/persistent_kernel.cuh)，只看 `MODE_OFFLINE` 的 `prepare_next_batch()`，按源码的五个步骤读：

1. 收尾上轮请求：把需要保留的 output_tokens 写入 tokens，推进 step；结束请求释放页。
2. 保存分页索引快照，供后续原地整理使用。
3. 整理仍活跃的请求，并在容量允许时加入新请求；准备 input_tokens 和 KV 页。
4. 清空未使用 slot，补齐索引数组末尾。
5. 更新页队列位置；若本轮有效 token 数为 0，返回 false。

`EVENT_END_OF_TASK_GRAPH` 的 scheduler 分支根据返回值决定继续下一轮 begin task，或者终止。图的重复执行与请求的重复提交不是一回事：offline 路径可在 GPU 内推进多轮，不要求 CPU 每个 token 重建任务图。

用一个无推测解码、单请求、prompt 长度 3 的纸面例子：

| 图轮次 | 本轮 input_tokens | 图结束时重点状态 |
|---|---|---|
| 首轮 prefill | p0,p1,p2 | step 前进到 3，末 prompt 位置的预测可存为 tokens[3]=q0 |
| 下一轮 decode | q0 | step 前进到 4，输出可存为 tokens[4]=q1 |
| 再下一轮 | q1 | 继续，或因 EOS/长度限制结束 |

假设本例 token 预算至少为 3、长度与页容量足够，且忽略 sampling 和特殊模式。若预算更小，prefill 可拆成多轮，不能套用这张一次处理完 prompt 的表。

`test_mode` 与 profiling 会改变这里的完成判断，因此用阶段 1 单轮测试不能证明多轮 serving 状态管理正确。看 trace 或输出 token 时，应先写下 mode、宏和实际处理轮数。

**8．把不变量与可变状态列出来**

| 信息 | 本次编译/运行中的典型行为 |
|---|---|
| 模型层结构、选定算子变体、切块 | 构图/编译时决定，跨轮复用 |
| 权重 | 推理时主要只读，地址仍需保持有效 |
| 中间激活 | 每轮按实际输入重写，多个层可能复用 buffer |
| KV cache | 按 token 位置和页管理不断更新 |
| step、request IDs、indptr | 随请求进度与批次整理变化 |
| event counters | 同一次运行里按图迭代累计 |

这张表把阶段 3 的“最近写入者”和阶段 4 的“同一描述符不同 iteration”连到真实模型：复用计算结构不等于复用上一轮数值。

**9．练习与自测**

不需要 GPU 的三份练习：完整 MLP shape 表；一层 Transformer 的 API/数据流图；一轮 prefill 与两轮 decode 的 step/token/KV 状态表。

运行扩展只选已经匹配硬件的随机权重 MLP 学习副本，先通过单段，再验证全子图。遵守阶段 1 的资源与输出目录安排；不用为本阶段自动下载 Qwen 权重或批量运行所有模型。

| 自测题 | 核对答案 |
|---|---|
| 合并 gate/up 的数学意义是什么？ | 将两个相同输入的线性投影组织到一次图中线性计算，输出仍保留两种特征 |
| `shuffle_tensors` 为什么影响 SiLU？ | 后继需按正确物理分组配对 gate/up |
| MLP 小测试含前置 RMSNorm 吗？ | 这条三步子图没有，完整 builder 另行添加 |
| 最终输出是否必然只是一层 argmax？ | 当前 greedy 路径有 partial/reduce；sampling 是另一组分支 |
| KV 页编号是否等于 token 位置？ | 不等，需经页表映射 |
| request ID 是否等于本轮 slot？ | 不一定，整理批次会改变 slot 与请求的对应 |
| 每个 token 是否要在 Python 中重新构图？ | 当前 offline 循环可由 GPU 推进并复用任务结构 |
| 为什么不能把 test mode 当作 serving 验证？ | 请求完成/轮次行为不同，覆盖面有限 |

完成这里，你已能串起单卡 MPK 的核心实现。继续阅读 [阶段 7](stage-7-guide-zh.md) 的多 GPU，或先进入 [阶段 8](stage-8-guide-zh.md) 学习如何检查性能证据。
