**MPK 阶段 1 具体导读：读通 RMSNorm 最小端到端测试**

这一阶段接续 [阶段 0](stage-0-guide-zh.md)，属于 [总阅读计划](reading-guide-zh.md) 的阶段 1。基于本地提交 `97de2592`，围绕一份约 90 行的测试，建立从 Python 张量到 GPU 计算结果的完整认识。

本阶段的目标：合上代码后，你能按顺序解释“数据在哪里、什么时候只是描述计算、什么时候生成代码、什么时候真正执行、结果如何拿到”。暂时不展开 C++ 依赖分析、scheduler 循环和 CUDA 归约细节；它们分别属于后续阶段。

主线可以只读源码完成。运行部分是环境就绪后的可选练习；本文编写时没有执行 GPU 测试或编译，文中的运行结果均为待观察项目，不是已测结果。

**1．先打开两个文件，建立阅读路线，10 分钟**

主文件：[test_rmsnorm_testmode.py](../../tests/runtime_python/test_mode/test_rmsnorm_testmode.py)。配套文件：[persistent_kernel.py](../../python/mirage/mpk/persistent_kernel.py)。

先通读测试一次，不跳进实现，给语句标上以下类别：

```text
准备数据
  torch.randn / torch.zeros
       ↓
创建构图与运行配置
  get_default_init_parameters → PersistentKernel(...)
       ↓
关联张量、添加计算
  attach_input → rmsnorm_layer
       ↓
生成代码、编译、加载和初始化
  compile
       ↓
执行并等待
  pk() → torch.cuda.synchronize()
       ↓
验证、清理
  torch_rmsnorm → max_diff → finalize
```

建议按以下顺序阅读，共约 160 分钟，不包括安装、编译和运行等待时间：

| 阅读内容 | 时间 | 这一段只解决的问题 |
|---|---:|---|
| 全局流程 | 10 分钟 | 测试有哪些阶段？ |
| 公式与三个张量 | 20 分钟 | 具体要算什么？ |
| 初始化与 test mode | 20 分钟 | 为什么一个算子也要 runtime 配置？ |
| 张量关联与构图 | 30 分钟 | 数据如何变成图里的对象？ |
| 编译产物 | 25 分钟 | `compile()` 做了哪些事？ |
| 执行与正确性检查 | 20 分钟 | 结果在哪里，测试证明了什么？ |
| 小练习与自测 | 35 分钟 | 能否独立复述整条路径？ |

源码跳转优先用符号，避免依赖行号：

```bash
rg -n 'def (__init__|get_default_init_parameters|_apply_test_mode_meta_defaults|attach_input|new_tensor|rmsnorm_layer|compile|__call__|finalize)' python/mirage/mpk/persistent_kernel.py
```

**2．先理解数学与数据，不急着看 MPK，20 分钟**

测试一开始定义了 PyTorch 参考实现：

```python
def torch_rmsnorm(x, weight, eps=1e-5):
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)
```

把输入的一行记作 `x[r, :]`，这一行有 `H` 个特征。它先计算平方的平均值，再将该行除以对应的均方根，最后乘逐特征权重：

```text
mean_square[r] = sum(x[r, j]^2, j=0..H-1) / H
out[r, j] = x[r, j] / sqrt(mean_square[r] + eps) * w[j]
```

这里局部变量虽然叫 `variance`，实际计算的是均方值，没有先减均值；不要因变量名把它理解成完整的方差或 LayerNorm。`eps` 是加在平方根内部的小正数。

手算一个两元素例子：取 `x=[3,4]`、`w=[1,2]`，仅为演示忽略 eps。均方值是 `12.5`，均方根约 `3.5355`，输出约为 `[0.8485, 2.2627]`。注意第二个元素最后还要乘权重 2。

测试中的真实规模是：

| Python 变量 | shape | dtype / 位置 | 作用 |
|---|---|---|---|
| `x` | `[16, 4096]` | BF16 / CUDA | 16 行待归一化的输入 |
| `w` | `[4096]` | BF16 / CUDA | 所有行共同使用的逐特征权重 |
| `out` | `[16, 4096]` | BF16 / CUDA | 预先分配的输出缓冲区，初始为 0 |

`mean(dim=-1, keepdim=True)` 沿最后一维做平均，将 `[16,4096]` 变成 `[16,1]`，这样每行的归一化因子可以广播到该行所有元素。权重 `[4096]` 则广播到所有行。

输入与输出各含 65,536 个 BF16 元素，各占 128 KiB；权重占 8 KiB。这里只统计这三个张量的有效数据，不包括 allocator、runtime、元数据和编译内存。这个例子不需要大模型权重，但并不意味着初始化或编译完全没有额外成本。

停下来回答：每行 RMSNorm 是否需要其他行的数据？如果不需要，图里能否把不同的行作为独立任务？

**3．构造 `PersistentKernel`：区分算子参数和 runtime 参数，20 分钟**

回到测试的初始化片段：

```python
num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
params = PersistentKernel.get_default_init_parameters()
params["test_mode"] = True
params["num_workers"] = num_workers
params["num_local_schedulers"] = num_schedulers
params["mpi_rank"] = 0
params["world_size"] = 1
pk = PersistentKernel(**params)
```

`params` 是字典，`**params` 将字典展开成命名参数。先看 `get_default_init_parameters()`，将默认值分成三组：

| 组别 | 关注字段 | 当前测试中的意义 |
|---|---|---|
| 运行方式 | `mode="offline"`、`world_size=1`、`mpi_rank=0` | 单 GPU 的 offline 路径 |
| 执行资源 | worker 数、本地/远程 scheduler 数 | 决定执行任务的 runtime 资源；测试覆盖了默认 worker 与本地 scheduler 数 |
| 推理状态 | 长度/请求/token/page 上限、`meta_tensors` | 满足 runtime 初始化与收尾需要；此处沿用最小配置 |

`get_configurations_from_gpu()` 在 [python/mirage/utils.py](../../python/mirage/utils.py) 中，按 SM 数选择 worker 数，再计算 scheduler 数。它是代码中的配置策略，不是性能自动搜索，也不代表任何 GPU 都一定受支持。此时读懂输入和返回值即可，不要自行把 worker 数改成全部 SM 数。

现在跳到 `PersistentKernel.__init__()`，只寻找以下几个动作：

1. 保存运行方式和资源参数，并将 `_is_compiled` 初始化为 `False`。
2. 创建 `self.kn_graph`，后续算子会添加到这个图中。
3. 从设备属性计算 `target_cc`，用于后续选择架构分支。
4. 在 `test_mode` 下调用 `_apply_test_mode_meta_defaults()`。
5. 检查所需元数据的 shape 与 dtype。

`_apply_test_mode_meta_defaults()` 会为缺失的 tokens、step、请求与分页索引等状态准备默认 CUDA 张量。所以“最小测试”依然使用实际 GPU runtime，不是 Python 模拟执行，也不是省略了 scheduler。

有两个容易混淆的细节。

**第一，算子的 16 行输入，不等于 serving 的 16 个请求。** 本测试没有把 `max_num_batched_tokens` 或 `max_num_batched_requests` 覆盖成 16，它们仍为默认值 1。这里 RMSNorm 的行数与切块来自 `x` 的 shape 和算子图；服务元数据用于给 runtime 提供最小可运行状态。不要将这种最小测试设置推广到依赖请求/token 索引的 attention 或完整模型。

**第二，test mode 的行为来自 Python 与 CUDA 两边。** Python 自动补元数据；`get_compile_command()` 添加 `-DMPK_TEST_MODE`；offline 的 `prepare_next_batch()` 在该宏下把已处理请求判为完成。本例默认只有一个请求，因此用于执行一轮测试图后结束。它不是“只做一个 task”，也不能泛化为任意多请求设置都只运行一轮。

本节练习：在笔记上画两列，左侧写 `x.shape=[16,4096]` 和 `grid_dim=(16,1,1)`，右侧写 worker/scheduler 数及 serving 元数据。分别回答“算哪些数据”和“由哪些执行者、在什么运行配置下执行”。

**4．从真实张量到图描述，再到一层 RMSNorm，30 分钟**

测试接下来调用：

```python
x_dt = pk.attach_input(x, name="x")
w_dt = pk.attach_input(w, name="w")
out_dt = pk.attach_input(out, name="out")
```

在 `attach_input()` 中，按这个顺序读即可：

```text
读取 torch tensor 的 shape、stride、dtype
  → 检查支持的布局
  → kn_graph.new_input(...) 创建 DTensor 描述
  → attach_torch_tensor(...) 关联现有张量
  → 保存名称映射和 Python tensor 引用
  → 返回 DTensor
```

`x` 是持有实际数据的 PyTorch 张量；`x_dt` 是 MPK 图里的描述对象。注册不会为 RMSNorm 计算一份输出，也不是把 x 拷贝成另一个独立的输入缓冲区。

代码还将张量存入 `_torch_tensor_refs`，以免 Python 对象提前被回收，导致生成/运行代码使用的 GPU 地址失效。对熟悉 C++ 的你，可以类比“保存了一个指针，就还必须保证所指对象活得足够久”。

为什么 `out` 也调用 `attach_input()`？这里这个接口负责把已有张量引入构图系统；张量在具体算子中是读入还是写出，由后续任务注册路径确定。不能把函数名中的 input 解读成“此张量永远只读”。

与之对照，`new_tensor()` 用 shape、dtype 和内存类别描述需要由 MPK 管理的张量，并注册相应分配方式。本测试已经有 PyTorch 的 `out`，这样运行结束后可以直接拿它比较结果。

接着看测试中的 layer 调用：

```python
block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
pk.rmsnorm_layer(
    input=x_dt,
    weight=w_dt,
    output=out_dt,
    grid_dim=(16, 1, 1),
    block_dim=block_dim,
)
```

上面为方便阅读，将原测试中变量 `batch_size` 展开为 16。现在跳到 `rmsnorm_layer()`，它的核心代码很短：

```python
tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
tb_graph.new_input(input, (0, -1, -1), 1, True)
tb_graph.new_input(weight, (-1, -1, -1), 0, True)
tb_graph.new_input(output, (0, -1, -1), 1, True)
self.kn_graph.customized([input, weight, output], tb_graph)
self.kn_graph.register_task(
    tb_graph, "rmsnorm_hopper" if self.target_cc >= 90 else "rmsnorm"
)
```

先不用展开 `CyTBGraph` 构造参数和末尾参数的每一个含义，只抓住两层描述：`kn_graph` 组织算子，`tb_graph` 描述这一层的线程块级工作划分与数据映射。

在这里，映射 tuple 的三个位置分别对应 grid 的 x/y/z 轴：`(0,-1,-1)` 让 grid.x 切分 tensor 的第 0 维；`-1` 表示不沿该 grid 轴切分。权重的 `(-1,-1,-1)` 表示每项任务都需要整份权重，而不是把权重复制成 16 个独立张量。

因此这层的普通计算任务可以画为：

```text
task 0 ：读取 x[0, :]  和 w[:] → 写 out[0, :]
task 1 ：读取 x[1, :]  和 w[:] → 写 out[1, :]
……
task 15：读取 x[15, :] 和 w[:] → 写 out[15, :]
```

这里有 16 项 RMSNorm 计算工作，但生成的任务表还会有开始、终止等控制任务；不能要求 JSON 总条数恰好等于 16。`block_dim` 是该算子任务的线程块配置，整个常驻 runtime 的启动维度另有逻辑。

这段构图代码没有直接写平方、求和、开方：它通过 `register_task()` 指定已有任务实现。当前 `target_cc >= 90` 选择 `rmsnorm_hopper`；文件名包含 Hopper 并不意味着只对恰好 `target_cc == 90` 生效，应看实际条件。

另一个看起来奇怪的地方是 `output` 也使用了 `tb_graph.new_input(...)`。当前手工注册路径把参数的 tile 描述放进 TBGraph，后面的 RMSNorm 注册函数按“前两个输入、最后一个输出”解释这些描述。本阶段记下这个接口约定即可，阶段 2 再追 C++。

本节练习：自己画出 `x / w / out` 与 `x_dt / w_dt / out_dt` 的关联，再画 16 项计算任务。必须标出权重共享、输出行互不覆盖，以及 worker 数与任务数是两回事。

**5．`compile()`：生成、编译、加载、初始化，25 分钟**

测试写的是：

```python
folder_path = os.path.dirname(__file__)
pk.compile(output_dir=folder_path)
```

进入 `compile()` 后，不要逐个研究 include 路径。沿下列关键语句定位：

| 关键语句/操作 | 做了什么 | 这一刻你应该问的问题 |
|---|---|---|
| `assert not self._is_compiled` | 检查当前对象尚未编译 | 能否直接对同一个对象重复调用 compile？ |
| `self.kn_graph.generate_task_graph(...)` | 交给底层生成任务图和 CUDA 文本 | 图描述如何变成可执行内容？先把 C++ 当作黑盒 |
| 写出 `results["json_file"]` | 保存任务、事件等结构 | 描述信息放在哪里？ |
| 写出 `results["cuda_code"] + HARD_CODE` | 保存生成代码与 Python 扩展入口 | Python 最终从哪个接口进入运行时？ |
| `subprocess.check_call(cc_cmd)` | 调用 nvcc，生成共享库 | 这里需要真实编译工具链 |
| `importlib` 加载 `.so` | 获得 init/launch/wait/finalize 等函数 | 编译产物怎样成为 Python 可调用对象？ |
| `self.init_func(...)` | 传入元数据/张量指针和运行配置，初始化 runtime | 编译方法不只是写磁盘文件 |
| `self._is_compiled = True` | 标记成功完成初始化 | 之后执行阶段可使用已加载入口 |

`compile()` 内的初始化可能涉及 GPU 内存分配和初始化 kernel，因此不能说“compile 完全不会动 GPU”。但目标 RMSNorm 的数值计算由后面的运行阶段触发。

本例是 offline 路径，编译工作文件先写到临时目录；给定 `output_dir` 后再复制保存产物。因此 `output_dir` 是保留产物的地方，不等于所有编译过程都只在该目录发生。

当前 rank 为 0，成功编译时会保存：

| 文件 | 用途 | 阶段 1 读到什么程度 |
|---|---|---|
| `task_graph_rank0.json` | 任务、事件、初始任务的描述 | 认出 `all_tasks`、`all_events`、`first_tasks` |
| `test_rank0.cu` | 生成的 CUDA 源码及扩展入口 | 搜索 `_execute_task` 和 `rms_norm` |
| `mpk_launcher_rank0.cpython-*.so` | 当前 Python/架构环境使用的编译模块 | 知道它由 Python 加载即可 |
| `kernel_metadata_rank0.json` | 用于后续加载兼容性检查的配置 | 与任务图 JSON 区分开 |

代码在 nvcc 编译之前就会复制 `.cu` 和任务图 JSON。**看到这两个文件存在，并不能证明编译成功，更不能证明数值正确。** 反过来，看到 `.so` 也不代表这一轮测试已经通过。

仅以结构示意一份任务图，下面不是生成文件的完整格式或真实输出：

```text
all_tasks：任务种类、变体、输入输出、相关事件……
all_events：事件类型、触发次数、任务范围……
first_tasks：初始执行入口……
```

本节练习：画出 `Python 图 → JSON/CUDA → nvcc → .so → init_func/launch_func`。给每条箭头标注执行者是 Python、C++ 还是外部编译器。生成后的 `.cu` 可能主要通过 include 引用 runtime；未在文件中搜到完整 scheduler 函数体并不奇怪。

**6．运行、读取结果与验证，20 分钟**

测试执行的是：

```python
pk()
torch.cuda.synchronize()
ref = torch_rmsnorm(x, w)
max_diff = (out - ref).abs().max().item()
```

`pk()` 调用 `PersistentKernel.__call__()`，后者取得当前 CUDA stream，将其转换为 launcher 接收的形式，再调用 `self.launch_func(stream_ptr)`。这一底层类的 `__call__()` 不会自动替你先执行 `compile()`；不要与更上层 `MPK` 包装类的行为混淆。

进一步的调用关系可先记为：

```text
Python pk()
  → PersistentKernel.__call__
  → 编译模块的 launch_func
  → C++ launch_persistent_kernel
  → GPU worker / scheduler 执行任务图
  → task 写入事先关联的 out 缓冲区
```

这里没有 `out = pk(x, w)`：输入输出已在构图时关联。`pk()` 没有返回一个新张量，测试仍然读取之前创建的 `out`。

在当前提交，`init_persistent_kernel()` 将 `split_worker_scheduler` 设为 `true`。因此本例沿当前代码默认走拆分 worker/scheduler 的路径，启动前还会执行准备 kernel。阶段 0 用单 persistent kernel 讲角色划分，是概念入口；读实际测试时，应按当前配置识别 launch 路径，而不能猜测只会看到一次 GPU kernel 启动。

`torch.cuda.synchronize()` 让测试在继续检查之前等待当前设备的 GPU 工作完成。这里还涉及 runtime 自己的 stream 与 event 衔接，阶段 4 再展开；先保留原测试的同步方式，不要根据 Python 调用返回就自行断言所有计算完成。

比较结果时，理解两点：

- `out[:2, :8]` 只用于打印局部样本；`max_diff` 对整个输出计算最大绝对差。
- `.item()` 把单个数值拿到 Python，用于判断 `max_diff < 0.05`。这不是性能计时。

此外，这个测试的参考实现和实际实现并非完全相同的浮点计算序列。参考函数默认 `eps=1e-5`；[task_register.cc](../../src/kernel/task_register.cc) 中两个 RMSNorm 注册函数给 CUDA 实现传入 `1e-6f`，而运算顺序与 BF16 舍入也会影响结果。因此原测试验证的是在当前随机输入、shape 和容差下结果接近，不能当作逐位等价证明，也不能把差异全部归因于 BF16。

如果在学习副本中进一步检查，可以使用与注册代码一致的 `eps=1e-6` 调用参考函数，并额外打印 FP32 下的误差：

```python
ref = torch_rmsnorm(x, w, eps=1e-6)
max_diff_fp32 = (out.float() - ref.float()).abs().max().item()
print("finite:", torch.isfinite(out).all().item())
print("max abs diff computed in fp32:", max_diff_fp32)
```

这只是让比较更容易解释，并不消除归约顺序和中间舍入差异。先记录原测试结果，再做这个扩展；出现失败时先定位原因，不要靠扩大容差让它通过。

最后读 `finalize()`：已编译的对象会调用底层清理函数，并设置终结标志。PyTorch 张量的生命周期仍由其引用管理；清理 runtime 不是重新计算或自动拷回输出。当前方法会检查是否已经 finalized，不要在同一个对象上随意重复清理。

**7．可选运行与产物观察**

如果当前 Mirage 和 CUDA 环境尚未就绪，先完成静态阅读即可。安装整个仓库不是通过本阶段的前置要求。

若准备运行，先确认运行所需条件，并注意这个脚本中的 `compile()` 会调用 nvcc。按资源约束，启动编译前查看逻辑 CPU 数、可用内存、系统负载和临时盘空间：

```bash
nproc
free -h
uptime
df -h /tmp
which nvcc
nvidia-smi
```

导入当前环境并查看模块来源，可以帮助发现“读的是本地源码，运行的却是另一份安装”的问题：

```bash
python - <<'PY'
import inspect
import torch
import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

print("mirage:", mirage.__file__)
print("PersistentKernel:", inspect.getfile(PersistentKernel))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

在仓库根目录，原测试的执行命令是：

```bash
python tests/runtime_python/test_mode/test_rmsnorm_testmode.py
```

它会把保留产物放在测试目录中。若要进行下面的修改练习，建议先复制一份到独立目录；因为测试使用 `os.path.dirname(__file__)`，产物也会保存在副本旁边：

```bash
mpk_stage1_dir=$(mktemp -d /tmp/mpk-stage1.XXXXXX)
cp tests/runtime_python/test_mode/test_rmsnorm_testmode.py "$mpk_stage1_dir/rmsnorm_reading.py"
python "$mpk_stage1_dir/rmsnorm_reading.py"
```

这两种运行方式任选一种，不需要为了开始阅读把它们都跑一遍。复制脚本不等于更换已安装的 Mirage，仍应先核对模块来源。

本例只启动一份测试编译，先不要并行运行多个变体。首次编译时观察 CPU 使用率、编译进程总 RSS 和 `/tmp` 增长。如果以后安装或重编译整个仓库，起步有效并发不超过 `min(16, 逻辑CPU数/4)`，并计入嵌套 NVCC 并发；模板编译内存压力大时进一步降低。当前 `get_compile_command()` 没有直接提供 `MAX_JOBS` 控制，不能假设设置它就一定限制了这里的 nvcc。

运行时按阶段观察，而不是只找一行 PASSED：

| 观察位置 | 应记录的内容 |
|---|---|
| 初始化 | 实际 GPU、worker/scheduler 数、`target_cc` |
| 编译开始 | nvcc 命令、是否含 `-DMPK_TEST_MODE`、产物目录 |
| 编译与初始化结束 | 模块是否成功加载，后续是否进入运行阶段 |
| 执行结束 | 输出片段、全量最大绝对差、通过/失败 |
| 清理结束 | 脚本是否正常退出 |

遇到导入失败、缺少 nvcc、架构/资源错误、数值不一致，分别回到对应阶段。缺少 JSON 往往说明图生成或更早阶段没完成；有 JSON 但没有 `.so` 则应优先看编译错误；不要把所有失败都视为 RMSNorm 数学错误。

若使用原测试目录，下面的只读片段可以概览任务类型。使用副本时，将 `artifact_dir` 改为实际输出目录：

```python
import json
from collections import Counter
from pathlib import Path

artifact_dir = Path("tests/runtime_python/test_mode")
graph = json.loads((artifact_dir / "task_graph_rank0.json").read_text())
print("top-level keys:", list(graph))
print("tasks:", len(graph["all_tasks"]))
print("events:", len(graph["all_events"]))
print("first tasks:", graph["first_tasks"])
print("task types:", Counter(t["task_type"] for t in graph["all_tasks"]))
```

把类型数值与 [runtime_header.h](../../include/mirage/persistent_kernel/runtime_header.h) 的 `TaskType` 对照，找出 RMSNorm 计算任务和控制任务。阶段 1 只观察描述，不手工修改 JSON 的依赖或任务类型。

**8．小练习与过关自测，35 分钟**

先做无需运行的练习：

1. 在测试每段代码旁标出：数据准备、初始化、构图、编译/加载、执行、验证、清理。
2. 写出三个真实 tensor 与三个 DTensor 的对应关系，标注 shape、dtype 和读写角色。
3. 给任意一项 RMSNorm 任务，例如第 7 行，写出它读取/写入的数据区域。
4. 画一张流程图，从 `torch.randn` 一直连到 `finalize()`，在 compile 旁列出四类保留产物。

环境就绪时可在学习副本中选做一个实验，先预测，再执行：

| 改动 | 先预测什么 | 需要观察什么 |
|---|---|---|
| 在创建随机张量前固定 `torch.manual_seed(0)` | 同一环境下输入更便于复查，但不因此证明逐位确定性 | 记录运行配置与误差 |
| 令 `w` 为全 1，保持 shape 不变 | 输出只体现均方根归一化 | 与采用相同 eps 的参考实现比较 |
| 把 `batch_size` 从 16 改为 8，保持 H、block 配置不变 | 输入/输出减少为 8 行，普通 RMSNorm 计算任务随之减少 | JSON 中 RMSNorm 任务数、数值结果；不要只比较总 task 条数 |

每次改图后新建对象重新编译，独立运行副本即可；不要对已经编译/清理的 `pk` 连续添加算子。本阶段不修改 hidden dimension、切块方式或 worker/scheduler 配比，它们涉及另外的实现约束。

最后合上源码，回答以下问题：

1. 输入 `[16,4096]` 的参考归一化因子 shape 是什么？为什么？
2. `x_dt` 与 `x` 的关系是什么？注册是否会立即执行 RMSNorm？
3. 为什么 `out` 也通过 `attach_input()` 注册？
4. 16 行输入、16 项普通计算任务、worker 数、serving 请求数，是否是同一个量？
5. 权重不沿 grid 切分，是否意味着创建了 16 份独立权重副本？
6. `test_mode=True` 是否意味着不走真实 runtime？本测试为何可以结束？
7. `compile()` 是否只调用 nvcc？其中的初始化是否可能使用 GPU？
8. 看到了任务图 JSON，能否确认编译和测试都成功？
9. `pk()` 从哪里获得输入输出地址？输出是否通过 Python 返回值取得？
10. 当前默认单卡路径会不会拆开 worker / scheduler？
11. 原测试 PASSED 能否证明逐位一致？eps 是否相同？
12. 阶段 2 应从哪一个 Python 调用继续追入 C++？

核对要点：

| 题目 | 参考答案 |
|---|---|
| 1 | `[16,1]`，最后一维平均且保留该维，供逐行广播 |
| 2 | `x_dt` 是关联实际张量的图描述；此时没有执行目标 RMSNorm |
| 3 | 接口负责引入现有张量；当前任务注册路径确定输入/输出角色 |
| 4 | 不是；前两者在这个切块配置下对应，worker 和 serving 元数据另有配置 |
| 5 | 不意味着物理复制；每项任务都可读取同一权重所对应的数据 |
| 6 | 仍走真实 runtime；补默认元数据并编入测试宏，本例单请求一轮后完成 |
| 7 | 还生成图/代码、保存产物、加载模块、初始化；初始化可能使用 GPU |
| 8 | 不能，JSON 在 nvcc 前已经保存，数值验证发生得更晚 |
| 9 | 构图关联并在初始化时传递张量信息/指针；结果写进既有 `out` |
| 10 | 当前初始化设 `split_worker_scheduler=true`，还有准备 kernel |
| 11 | 不能；是特定输入下的容差检查，参考 eps 为 `1e-5`，注册调用为 `1e-6` |
| 12 | 一条从 `rmsnorm_layer → register_task` 追注册；另一条从 `compile → generate_task_graph` 追代码生成 |

能不用看测试，复述数据、图描述、编译模块与运行结果之间的关系，就可以进入阶段 2。不需要在此时读懂 RMSNorm 的 CUDA 归约循环，也不需要解释完整的事件生成算法。

继续阅读：[阶段 2 导读 从 Python 构图到生成的 CUDA](stage-2-guide-zh.md)。
