# 5 自由度纤维梁 PINN / E-PINN

本工程读取 `E:\MDOF_case` 中 MATLAB 已生成的荷载、结构矩阵、Steel02
纤维参数和 ETDM 核。Python 不重新生成波浪荷载，也不重新组装 `M、C、K0`
或 ETDM 权重。

## 数据文件

- `wave_loads_300.mat`：`Fwave、time、M、C、K0` 及纤维梁参数。
- `wave_responses_newmark_300.mat`：MATLAB 响应，只供最终精度评估，不参与训练 loss。
- `wave_responses_etdm_300.mat`：固定的 `ETDM/A` 和参考响应。

## 网络与 loss

PINN 和 E-PINN 均采用三层 LSTM（层间无 ReLU），随后连接
`FC -> ReLU -> FC`。原单位波浪荷载首先进入权重固定的 SCL，按初始刚度
得到弹性响应。E-PINN 仍以弹性位移增量为输入并直接输出弹塑性位移增量。
PINN 改为以弹性位移全时程为输入，并直接输出弹塑性结构的总位移全时程；
PINN 的输入和输出分别用固定的 `0.5 m` 尺度转换，不采用数据集 RMS 缩放。

- PINN：采用二阶中心差分由预测位移计算速度和加速度；TBPTT分段末点因未来
  位移尚不可用而采用二阶向后差分。随后将预测位移历史送入 Steel02 纤维梁
  得到非线性恢复力 `R=Fint-K0*u`。唯一的训练目标为
  `mean((inv(K0)*(M*a + C*v + K0*u + R - Fwave))^2)`。该形式将完整的
  运动方程残差左乘固定的初始刚度逆矩阵，使残差具有位移单位。
- E-PINN：由预测位移计算非线性恢复力并送入固定 SCL；唯一的训练目标为
  `mean((du_LSTM - du_SCL)^2)`。

以上两个 loss 都不使用标签 loss、响应全量 loss、额外权重或数据集 RMS
缩放。PINN 和 E-PINN loss 均为物理位移残差的 MSE，单位为 `m^2`；但两者
对应的物理残差定义不同，因此数值大小仍不能直接作为精度的横向比较。

SCL 的 `ETDM/A` 使用 `register_buffer` 保存，不进入优化器；梯度可以穿过
SCL 回传到 LSTM，但不会修改 SCL 权重。

PINN 的一个完整响应批次采用同一套网络参数计算。TBPTT 分段只用于限制反向
传播长度，各段梯度累计完成以后才执行一次优化器更新，避免同一条 Steel02
滞回路径在中途切换网络参数。

## 安装与静态检查

```bash
pip install -r requirements.txt
python static_check.py --data-root ..
```

`ninja` 已列入依赖，用于编译 Steel02 C++/CUDA 扩展。终端显示
`Steel02 CUDA extension loaded successfully.` 即代表扩展加载成功。

静态检查验证 MATLAB 数据维度、Newmark 运动学、Steel02 恢复力及切线、
ETDM/SCL 重构、两种模型的前向/反向和 TBPTT 状态连续性，不执行优化器更新。

## 训练

工程提供三种彼此独立的训练/预测方式：

- `PINN_MDOFSys_Train.py` / `PINN_MDOFSys_Predict.py`：全量位移PINN；
- `EPINN_MDOFSys_Train.py` / `EPINN_MDOFSys_Predict.py`：位移增量E-PINN；
- `EPINN_MDOFSys_Full_Train.py` / `EPINN_MDOFSys_Full_Predict.py`：新增的
  全量位移E-PINN。其输入为弹性全量位移，LSTM输出弹塑性全量位移，唯一
  loss为`mean((u_LSTM-u_SCL)^2)`，单位为`m^2`。

先用短序列检查运行环境：

```bash
python PINN_MDOFSys_Train.py --data-root .. --epochs 1 --batch-size 2 \
  --sequence-length 64 --tbptt-length 64 --hidden-size 8 --fc-size 8

python EPINN_MDOFSys_Train.py --data-root .. --epochs 1 --batch-size 2 \
  --sequence-length 64 --tbptt-length 64 --time-truncation 64 \
  --hidden-size 8 --fc-size 8
```

正式训练：

```bash
python PINN_MDOFSys_Train.py --data-root .. --epochs 1000 \
  --batch-size 10 --tbptt-length 500

python EPINN_MDOFSys_Train.py --data-root .. --epochs 1000 \
  --batch-size 10 --tbptt-length 500

python EPINN_MDOFSys_Full_Train.py --data-root .. --epochs 1000 \
  --batch-size 10 --tbptt-length 500
```

### 四种时序拼接对照

全量入口新增 `--labelled-samples 10 --label-weight 0.1`：指定10条训练样本
参加全量位移标签MSE。默认标签数为0以保留旧主线。CSV分别记录物理MSE、
标签MSE、连续性MSE，检查点保存标签索引和网络参数量。
`run_full_stitch_comparison.py --data-root .. --output-dir comparison_run`
按统一设置顺序训练其余三组（不重复LSTM-hidden），每组200 epochs，随后
自动测试并保存MAT结果。输出目录必须尚不存在，状态写入`status.json`。

增量与全量 E-PINN 训练入口均支持 `--sequence-variant` 一个开关，原有命令
不加参数时仍为 `lstm-hidden`，因此旧主线不变：

| `--sequence-variant` | 对照形式 |
|---|---|
| `lstm-hidden` | LSTM 隐状态跨段传递 |
| `transformer-hidden` | Transformer 因果记忆跨段传递 |
| `lstm-explicit-overlap` | LSTM 显式初态、边界重叠及连续性 loss |
| `transformer-explicit-overlap` | Transformer 显式初态、边界重叠及连续性 loss |

例如，对增量 E-PINN 运行老师提出的严格重叠形式：

```bash
python EPINN_MDOFSys_Train.py --data-root .. --epochs 1000 \
  --batch-size 10 --tbptt-length 500 --hidden-size 120 --fc-size 120 \
  --sequence-variant transformer-explicit-overlap \
  --continuity-loss-weight 1.0 --transformer-layers 3 \
  --transformer-heads 4 --transformer-memory-length 128
```

将脚本名替换为 `EPINN_MDOFSys_Full_Train.py` 即可对全量 E-PINN 做同一
组对照。显式模式对每段的 `N` 个物理时刻在网络内部加入一个初态 token，
得到 `N+1` 个网络输出；第一个输出与上一段末状态构成连续性 loss，随后被
丢弃，余下 `N` 个输出进入 Steel02/SCL，因此保存的最终时程没有重复点。
Steel02 和 SCL 的历史仍在段间连续传递，但不作为网络输入。

四组合静态检查：

```bash
python static_check_stitch_variants.py --data-root ..
```

测试非默认组合时可直接指定两个开关以自动选择相应的最新训练目录：

```bash
python EPINN_MDOFSys_Test.py --variant increment --data-root .. \
  --sequence-variant transformer-explicit-overlap
```

底层的 `--sequence-model` 与 `--stitch-mode` 两个参数仍保留，便于单独控制；
一旦给出 `--sequence-variant`，它会覆盖这两个底层参数。

PINN 和 E-PINN 均默认采用全局梯度范数裁剪 `1.0`。E-PINN 会在 CSV 中记录
裁剪前的逐轮平均梯度范数 `train_gradient_norm_before_clip`；可通过
`--gradient-clip 0.5` 调小阈值，或通过 `--gradient-clip 0` 关闭裁剪。

每轮的训练和验证 loss 写入时间戳目录中的 `epoch_loss.csv`，并更新
`epoch_loss.png`。每种方法只保留验证 loss 最小的 10 个 checkpoint。

## 预测

架构已经改变，PINN 不能加载此前任何位移增量输出架构的 checkpoint。

```bash
python PINN_MDOFSys_Predict.py checkpoint.pth --data-root ..
python EPINN_MDOFSys_Predict.py checkpoint.pth --data-root ..
python EPINN_MDOFSys_Full_Predict.py checkpoint.pth --data-root ..
```
# Representative labelled training samples

## Fixed per-DOF scaling (increment E-PINN)

New increment E-PINN training defaults to `--dof-scaling stiffness-profile`.
`--dof-scaling uniform` retains the former scalar scales. Full E-PINN and PINN
scaling is unchanged by this option. The fixed reference shape is
`v = solve(K0, ones(5))`, `r = v / v[-1]`, using the imported MATLAB stiffness.
The equal unit nodal forces define a reference shape only; actual wave loading
and the physical equations are not changed. No sample normalization/RMS is used.
The increment and displacement scales are `0.1*r` and `0.5*r` metres.
Input elastic increments are divided by the increment vector; raw network outputs
are multiplied by the same vector before constitutive evaluation and SCL. All
increment consistency/label losses use this same increment vector; local
cumulative losses use the displacement vector. Physical outputs remain in metres.
These scales define a relative weighting, not a guarantee of equal dynamic
responses or accurate drift. Metadata stores the exact vectors. Test using
`EPINN_MDOFSys_Test.py --variant increment`, which supports both scalar legacy
checkpoints and new vector scales; the older Predict entry point rejects vectors.

For the current K0 the bottom-to-top increment scales are
`[0.0066666667, 0.0235, 0.0465, 0.0726666667, 0.1]` m and displacement scales are
`[0.0333333333, 0.1175, 0.2325, 0.3633333333, 0.5]` m.

Increment E-PINN, full E-PINN and full PINN training accept
`--label-selection representative|random` (new training default: `representative`).
Use `--labelled-samples 10` to keep the label budget at ten for each method.
The train/validation/test split is unchanged. Only training samples are eligible.
Selection covers log(1 + maximum fiber ductility) and signed mean top displacement
over the second half of the reference history. The latter is an offset diagnostic,
not a definition of unloaded residual displacement. After seeding the weak/strong
ductility and negative/positive offset extremes, deterministic farthest-point
selection fills the remaining slots using training-only min-max feature ranges.
This is response-informed selection from already computed MATLAB training data;
it is not random sampling or label-free active learning.

The strategy and exact indices are saved in the existing training configuration
and checkpoint. Legacy configurations without this field retain random selection.
For the current 300-sample dataset, ten representative MATLAB sample indices are
22, 47, 48, 67, 76, 111, 128, 145, 180, 289. They are computed, not hardcoded.
Increment E-PINN label supervision remains displacement-increment MSE (weight 0.2)
plus 32-step local cumulative-error MSE (weight 0.01); no loss weights or physical
branches are changed by this selection option.
