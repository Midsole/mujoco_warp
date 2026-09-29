# 稠密 SDF 与梯度缓存实验

`dense_sdf.py` 把 MuJoCo 编译得到的原生网格八叉树预采样为规则网格，每个格点保存 float32 的 `(distance, gx, gy, gz)`。两个手指和方块均使用原来的网格、位姿与物理参数；方块没有替换成解析 SDF。

运行时直接计算数组索引，对八个相邻格点进行三线性插值。梯度不预先归一化，接触搜索与最终法向构造仍由 MuJoCo Warp 完成。体积之外保留原实现的包围盒延拓和有限差分梯度。缓存接入碰撞路径；传感器等其他原生八叉树调用不切换。

提供两种梯度缓存：`gradient_mode="vertex"` 保存格点梯度并插值；`gradient_mode="cell"` 额外为每个单元保存 12 个 float32 梯度系数，运行时直接计算该单元内距离插值的导数。后者避免跨越单元边界混合不连续的梯度，但消耗更多显存。它没有在运行时重新遍历八叉树或用距离有限差分求体积内部梯度。

## 使用

```python
import mujoco_warp as mjw
from dense_sdf import attach_dense_sdf
from grasp import build_model, make_trajectory, validate
from grasp_warp import rollout_warp

model = build_model()
warp_model = mjw.put_model(model)
metadata = attach_dense_sdf(model, warp_model, resolution=257, gradient_mode="cell")
trajectory = make_trajectory(model)
trace = rollout_warp(model, trajectory, warp_model=warp_model)
print(metadata, validate(model, trajectory, trace))
```

在创建 CUDA Graph **之前**构建缓存。切换缓存后必须重新捕获 graph；旧 graph 保留捕获时的参数。缓存只适用于构建它的模型及设备；几何数据发生变化后应重新构建。

执行 graph 时必须保持对应缓存数组存活。若要交替使用旧、新两个 graph，需分别保留它们的缓存引用，不能只覆盖 `warp_model.dense_sdf` 后丢弃旧缓存。

网页默认运行使用 257³、cell 模式稠密 SDF，也可选择八叉树对照；原生命令行仍使用八叉树基线。
缓存附加在 Warp model 上，所有仿真环境共享，不随环境数量复制。目前不支持 C++ 插件 SDF 或多套不同的批量 mesh 数据。

## 复现

从仓库根目录运行：

```sh
uv run contrib/piper_h/dense_sdf_benchmark.py --resolutions 65 129 257 --gradient-mode vertex --validate
uv run contrib/piper_h/dense_sdf_benchmark.py --resolutions 129 257 --gradient-mode vertex --worlds 1 512 --repeats 3 --output contrib/piper_h/results/dense_sdf/full_benchmark.json
uv run contrib/piper_h/dense_sdf_benchmark.py --resolutions 257 --profile --repeats 3 --output contrib/piper_h/results/dense_sdf/profile.json
uv run contrib/piper_h/dense_sdf_benchmark.py --resolutions 257 --gradient-mode cell --validate --profile --worlds 1 512 --repeats 3 --output contrib/piper_h/results/dense_sdf/cell_benchmark.json
uv run pytest contrib/piper_h/dense_sdf_test.py -q
```

查询测试每个网格使用固定种子的 65,536 个体积随机点，覆盖包围盒之外的延拓；查询距离与梯度一起计时，预热后使用 CUDA events 测量 5 组、每组 100 次 graph 执行。重复点集可能受 GPU 缓存影响，不能把查询加速倍数直接当作仿真加速倍数。

完整仿真使用相同的 24,001 步控制轨迹，八叉树/稠密版本交替执行；计时包含 CPU 提交与 GPU 执行，排除模型编译、缓存构建、graph 捕获和预热。批量运行检查末态有限性、容量溢出和落点；单环境完整 trace 另检查搬运高度、双指接触和稳定性。碰撞分项使用 graph 内 CUDA events，每 20 步采样一次，以所代表的步数加权求平均。

512 环境使用相同初态和控制，环境间可能有细微数值差异；这项测试没有覆盖随机初态或不同策略产生的分散查询负载。

## 精度与显存限制

`vertex` 模式的距离与梯度分别插值，因此插值梯度不保证等于插值距离的精确导数。网格叶边界、薄壁和棱边附近的梯度可能变化剧烈。抓取通过仅验证本例轨迹，不能证明所有接触位置都等价。报告同时记录所有随机点与表面距离小于 1 mm 的点的误差，并单独计数无法计算方向角的零梯度点。`cell` 模式避免这种独立插值误差，但距离网格本身仍可能存在采样误差。

257³ 在本例中能重现深度 8 八叉树的距离值至 float32 舍入误差量级，但梯度仍是近似。129³ 内存更少，部分手指位置的距离误差可达约 0.46 mm。实验保留原八叉树用于对照和其他查询，因此以下是**额外缓存**，不是模型总显存：

| 每个网格的分辨率 | 三个网格额外缓存 |
| --- | ---: |
| 65³ | 12.57 MiB |
| 129³ | 98.27 MiB |
| 257³ | 777.04 MiB |

上表为 `vertex` 模式。推荐的 `cell` 模式在 257³ 下额外缓存合计 3.01 GiB（含当前实现保留的格点数据），所有环境共享。只在显式调用缓存构建函数时分配；该函数默认采用 257³、`cell` 模式。

结果文件位于 `results/dense_sdf/`，完整原始数据保留各次计时及精度分位数。

## 本次验证结果

设备：RTX 5090 D v2；MuJoCo 3.13.1，Warp 1.15.0。模型为 60 mm 方块与两个 SDF 手指；距离场来自同一深度 8 的网格八叉树，没有解析方块。

### 整段仿真

每次 24,001 步 × 1 ms；每组交替测量三次，表中为中位数。排除编译、上传、缓存构建、graph 捕获和预热，包含 CPU 提交与 GPU 执行。512 环境使用相同初态及控制。

| 缓存 | 环境数 | 额外显存 GiB | 八叉树 s | 稠密 s | 耗时减少 | 稠密 ms/批次物理步 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vertex 129³ | 1 | 0.096 | 19.406 | 10.995 | 43.3% | 0.4581 |
| vertex 129³ | 512 | 0.096 | 31.896 | 16.829 | 47.2% | 0.7012 |
| vertex 257³ | 1 | 0.759 | 19.406 | 10.951 | 43.6% | 0.4563 |
| vertex 257³ | 512 | 0.759 | 31.898 | 16.799 | 47.3% | 0.6999 |
| cell 257³ | 1 | 3.009 | 19.417 | 11.000 | 43.3% | 0.4583 |
| cell 257³ | 512 | 3.009 | 33.263 | 17.101 | 48.6% | 0.7125 |

显存为三个网格合计额外缓存，由所有环境共享。当前实现仍保留原八叉树。两个版本均无溢出，所有计时运行的最大落点误差小于 5 mm。单环境完整轨迹另验证搬运高度、双指接触与最终稳定性；三种 vertex 分辨率及 cell 257³ 均通过。

vertex 数据来自增加 cell 支持前的实现；cell 数据来自最终实现。两者分别与同一轮实验中的八叉树分支交替对照，不以跨批次差值判断两种缓存的性能优劣。

### 推荐版本：cell 257³ 的碰撞分项

在原 CUDA graph 中每 20 步采样 CUDA events，按采样所代表的步数加权。选择碰撞平均耗时中位数对应的完整一轮，三个阶段不重叠。此处未另跑分项计时扰动校准；端到端结论以上表未插入 events 的运行结果为准。

| 阶段 | 八叉树 ms/步 | 占碰撞比例 | 稠密 ms/步 | 占碰撞比例 |
| --- | ---: | ---: | ---: | ---: |
| 粗筛 | 0.003228 | 0.69% | 0.003229 | 2.74% |
| 精细碰撞 | 0.465273 | 98.98% | 0.113054 | 95.92% |
| 清零与其他 | 0.001579 | 0.34% | 0.001578 | 1.34% |
| 碰撞合计 | 0.470080 | 100.00% | 0.117860 | 100.00% |

### 固定姿态碰撞

两版使用之前保存的相同夹紧/搬运姿态，冻结状态，仅重复 collision；每组三次，每次 500 次查询。下表为 wall time 中位数。

| 姿态 | 环境数 | 八叉树 ms/次 | 稠密 ms/次 | 接触点/环境（两版相同） |
| --- | ---: | ---: | ---: | ---: |
| grip | 1 | 0.514131 | 0.119024 | 6 |
| grip | 512 | 0.870564 | 0.224109 | 6 |
| carry | 1 | 0.668096 | 0.142642 | 8 |
| carry | 512 | 1.088559 | 0.243359 | 8 |

### 查询精度与速度

cell 257³：每个网格使用固定随机种子的 65,536 个点，同时查询距离与梯度。包含体积外查询。预热后的 CUDA event 测量可能受重复查询的缓存命中影响。角度在 CPU 上用 float64 计算，排除零梯度；“表面附近”指原距离绝对值 < 1 mm。

| 网格 | 查询加速 | 距离最大误差 m | 表面附近梯度角最大误差 ° | 全体点梯度角最大误差 ° |
| --- | ---: | ---: | ---: | ---: |
| gripper_link1_mesh | 2.57× | 2.61e-08 | 0.006506 | 14.388 |
| gripper_link2_mesh | 2.71× | 2.71e-08 | 0.006947 | 0.021 |
| grasp_cube | 3.18× | 9.31e-09 | 0.001533 | 0.337 |

vertex 模式会跨单元插值不连续的梯度。257³ 的手指表面附近梯度角 P95 仍约 41–44°，因此推荐 cell 模式；它预存每个单元内距离多项式的梯度系数，额外使用约 3.01 GiB。

cell 模式的表面附近误差很小，但全体采样点仍有梯度离群值，不能声称与八叉树在全空间严格等价。方块中有 183 个点因零梯度不参与角度统计。当前结论针对这一模型、分辨率与轨迹，不覆盖随机化批量环境。

cell 257³ 的预采样与梯度系数构建在已有编译缓存下约 16 ms；这不包含 MuJoCo 编译原八叉树或 Warp 首次 JIT。

测量时保留桌面及已有仿真服务；进程采样未观察到其他服务使用 GPU 计算单元，未强制独占设备。

原始数据：`full_benchmark.json`（vertex）、`cell_benchmark.json`（cell）、`validation.json`、`summary.json`。

后续三版本统一复测及代码审查见 [REVIEW_20260929.md](REVIEW_20260929.md)。
