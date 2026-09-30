**MPK 阶段 2 导读 从 Python 构图到生成的 CUDA**

承接 [阶段 1](stage-1-guide-zh.md)，这次继续追 RMSNorm，但把 `compile()` 的黑盒打开。目标是解释一行 `pk.rmsnorm_layer(...)` 怎样对应到生成代码中的一次 CUDA device 函数调用。基于提交 `97de2592`；不需要重新编译，已有阶段 1 产物时可对照阅读。

建议用 3～4 小时分两次完成：第一次追注册，第二次追代码生成与指针。不要从 `runtime.cc` 第一行一直读到最后。

**1．先区分运行地点和发生时间**

| 代码 | 在哪里执行 | 做什么 |
|---|---|---|
| Python `rmsnorm_layer()` | CPU，构图时 | 描述工作划分、注册算子 |
| Cython `.pyx` 桥接 | CPU，构图/编译时 | 转换参数并调用 C++ |
| C++ `TaskRegister`、`Graph::generate_task_graph()` | CPU，代码生成时 | 选择实现，构造任务描述和 CUDA 文本 |
| nvcc | CPU，编译时 | 编译生成的 CUDA 与头文件 |
| 生成模块的初始化函数 | host 端初始化阶段 | 绑定地址，准备 runtime 状态 |
| `_execute_task()`、`rms_norm_hopper_impl()` | GPU，执行时 | 分派任务并计算结果 |

“用 C++ 写成”不等于“在 GPU 上运行”。本阶段最重要的辨别方法是看：这段代码是在操作一个 `TaskDesc`，还是在拼出将来操作 `TaskDesc` 的字符串？

**2．追第一条链 注册一种工作**

按这个顺序打开文件，始终跟着 RMSNorm 分支：

```text
persistent_kernel.py：PersistentKernel.rmsnorm_layer
  → kernel.py：KNGraph.customized / register_task
  → _cython/core.pyx：CyKNGraph.customized / register_task
  → src/kernel/graph.cc：Graph::register_task
  → src/kernel/task_register.cc：register_rmsnorm_hopper_task
  → register_task_variant
```

入口文件：[persistent_kernel.py](../../python/mirage/mpk/persistent_kernel.py)、[kernel.py](../../python/mirage/kernel.py)、[core.pyx](../../python/mirage/_cython/core.pyx)、[graph.cc](../../src/kernel/graph.cc)、[task_register.cc](../../src/kernel/task_register.cc)。

阶段 1 已经看到 `TBGraph` 的三个参数描述：input、weight、output。`customized()` 将这份线程块级描述挂到 kernel 图的一个 customized 节点上；随后 `register_task()` 给它指定任务实现。

这里有一个值得亲自核对的接口细节：Python 接口带有 `bgraph`，但当前 Cython `register_task()` 最终调用的是 `self.p_kgraph.register_task(cname, cparams)`；C++ `Graph::register_task()` 使用 `operators.back()`，并检查它是 customized 节点。所以这条路径依赖“先添加 customized 节点，再紧接着注册任务”的调用顺序。不能仅凭 Python 参数表，认为 C++ 会按传入 bgraph 在所有节点里任意查找。

在 `graph.cc` 的 `rmsnorm_hopper` 分支，你会看到类似：

```cpp
task_config[op] = std::make_tuple(2, 1, TASK_RMS_NORM_HOPPER, variant_id);
```

四项依次是输入数、输出数、任务种类、实现变体。它回答了阶段 1 的疑问：虽然 output 的 tile 也通过 `tb_graph.new_input()` 描述，任务注册仍将前两个参数作为输入，最后一个作为输出。

练习：在一张纸上写出 `x_dt / w_dt / out_dt` 的参数顺序，并给这个 tuple 的四个位置各写一句解释。不要把 `variant_id` 当成具体某一行的 task ID。

**3．读懂生成一个调用的 C++**

在 `register_rmsnorm_hopper_task()` 中，只跟下面四步：

1. 从 `bgraph.operators` 按约定拆出两个输入与一个输出。
2. 读取输出 tile 的 `batch_size` 和 `hidden_dim`。
3. 使用 `CodeKeeper` 写出 device 函数调用文本。
4. 将该文本交给 `register_task_variant()`，取得变体编号。

阶段 1 的整张输入是 `[16,4096]`，沿第 0 维切成 16 份，因此这里用于该任务的 tile 形状是 `[1,4096]`。不要把注册函数里的局部 `batch_size` 直接理解成整个 PyTorch batch。

假设选择 Hopper 分支，生成调用应具有如下形式。下面是根据源码手工展开的示意，不是本次运行输出：

```cpp
kernel::rms_norm_hopper_impl<bfloat16, 1, 4096>(
    task_desc->input_ptrs[0],
    task_desc->input_ptrs[1],
    task_desc->output_ptrs[0],
    1e-6f);
```

`bfloat16`、`1`、`4096` 是模板实参；具体张量地址来自运行时描述符。设备函数的 `NUM_THREADS` 在该调用中没有显式传入，而使用定义中的默认值。这也是为什么不能随意修改 Python 的 block 配置，期待所有底层模板自动同步改变。

`CodeKeeper` 的 `code.e("... $ ...", value)` 是生成文本的工具。本阶段只需把 `$` 代入，就能读懂输出；不必通读通用 transpiler。

继续看 `register_task_variant()`：它在该 task type 的代码字符串列表中查找相同字符串，找到就复用编号，否则追加。它不是在 GPU 上做 autotuning，也不必为每一行 RMSNorm 分配一个新 variant。16 项任务可以共享同一份计算实现，只是数据地址不同。

练习：判断“把 16 行改成 8 行且仍每行一项任务”是否必然产生不同的 RMSNorm 代码变体。答案是不必然：tile 仍是 `[1,4096]`，生成调用文本可以相同；总任务数量则改变。具体编号取决于注册表状态，不要把某次的数字当成稳定接口。

**4．追第二条链 生成完整任务图和分派器**

现在从 `compile()` 重新进入：

```text
PersistentKernel.compile
  → KNGraph.generate_task_graph
  → CyKNGraph.generate_task_graph
  → Graph::generate_task_graph
      register_mugraph
      sanity_check
      print_task_graph
  → 返回 cuda_code / json_file 两段文本
```

核心在 [src/kernel/runtime.cc](../../src/kernel/runtime.cc)。先找 `Graph::generate_task_graph()`，认出三个容器 `all_tasks`、`all_events`、`first_tasks`；本阶段把 `register_mugraph()` 的依赖算法留给阶段 3。

`print_task_graph()` 的名字容易让人以为只是打印日志，实际上它生成初始化/加载代码、任务图 JSON、`_execute_task()` 等内容。Cython 将 `TaskGraphResult` 的两个字符串转换成 Python 字典，`compile()` 再将它们写入文件。

接着搜索 `_execute_task`。当前生成器遍历 `all_task_variants`，为 task type 与 variant 的组合生成 `if / else if` 分支，而不是为每个任务生成一个独立 `__global__` kernel。

```text
task_type + variant_id
  → 选中一段已注册代码
  → 从 TaskDesc 取指针
  → 调用 CUDA device 函数
```

这一步将“有哪些任务实例”和“有哪些计算实现”分开了。以后读大型模型时，先统计任务数，再统计实现变体数，二者不应混用。

**5．跟踪第 7 行的地址如何传下去**

以连续 BF16 张量 `x[16,4096]` 为例，第 7 行的元素偏移是 `7×4096=28672`，字节偏移是 `57344`。权重始终从第 0 个元素开始，输出同样偏移到第 7 行。

在 `print_task_graph()` 的输入/输出序列化代码中找 `offset`、`input_map`、`stride`、`get_datatype_size`。它依据 tile 位置和原 tensor stride 计算偏移，并把字节偏移写到 JSON。对于这份普通张量，预期关联关系是：

| 当前任务的参数 | 基对象名称 | 字节偏移 |
|---|---|---:|
| 输入激活 | `x` | 57344 |
| 权重 | `w` | 0 |
| 输出激活 | `out` | 57344 |

这是按行号推导的结果，不意味着“第 7 行”一定在 `all_tasks[7]`：任务表还包含控制项，复杂图还会重排。

再找生成的 `construct_task_graph()`：它从 JSON 读名称与 offset，通过 `all_tensors` 找到本次初始化传入的基地址，执行 `static_cast<char*>(base) + offset`。因此 JSON 里的 `base_ptr` 在这个路径中是名称字符串，不是可跨进程使用的原始 GPU 地址。

`FullTaskDesc` 携带较完整的 tensor 信息；初始化阶段将其转换成供 worker 使用的 `TaskDesc`。到 `rms_norm_hopper_impl()` 时，输入指针已经定位到任务 tile。函数内的 `batch_idx * HIDDEN_DIM` 是 tile 内的行偏移，不能再加一遍原始全局行号。

练习：把第 7 行改成第 15 行，手算输入/输出偏移。再说明为什么对于有 padding 的张量，应按真实 stride 算，不能一律用逻辑 `hidden_dim` 代替行跨度。

**6．用产物反向验证**

若已有阶段 1 产物，按顺序做四件事：

1. 在 `test_rank0.cu` 中搜索 `_execute_task` 和 `rms_norm_hopper_impl`，对照模板参数。
2. 在任务图 JSON 中找到 RMSNorm 类型的 task，查看 `inputs`、`outputs` 中的名字、offset、dims、strides。
3. 从 JSON 中任选一项，依据指针偏移推回它对应哪一行，而不是猜 task ID。
4. 回到注册函数找到生成这段调用的 `code.e()`。

只读观察命令，路径按实际产物目录替换：

```bash
rg -n '_execute_task|rms_norm|construct_task_graph' tests/runtime_python/test_mode/test_rank0.cu
```

没有产物时，手工展开调用文本与偏移表同样能完成本阶段。要重新运行时沿用阶段 1 的资源检查与单测试编译方式，不在这里另起批量编译。

**7．自测与过关标准**

| 自测题 | 核对答案 |
|---|---|
| 哪段 C++ 在生成 CUDA，哪段在 GPU 上执行？ | `TaskRegister/runtime.cc` 拼文本；生成的 device 分派器和算子在 GPU 上执行 |
| `register_task()` 当前绑定哪个图节点？ | C++ 使用刚添加的 `operators.back()` customized 节点 |
| 为什么 16 项 RMSNorm 可以只有一份 variant？ | 相同 tile 计算代码可复用，不同任务带不同地址 |
| task type 与 variant 各解决什么问题？ | 前者区分任务种类，后者区分该种类下的生成实现 |
| 生成 JSON 是否已经保存了 tensor 的全部数值？ | 没有；是结构、名称与偏移等描述 |
| offset 的单位是什么？ | 这里序列化并按 `char*` 相加的是字节偏移 |
| 能否把 output 的 `new_input` 字样当成只读证明？ | 不能，要看 task_config 的参数角色与注册函数 |
| 改 shape 是否只需修改 JSON？ | 通常不行；模板实参、布局、任务划分和编译模块也可能需要变化 |

交付给自己的笔记应有两张调用链、一份手工展开的 RMSNorm 调用、一张地址偏移表。下一步读 [阶段 3](stage-3-guide-zh.md)，解释这些 task 之间的依赖如何生成。
