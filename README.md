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

- PINN：对预测位移作差得到位移增量，用 Newmark 平均加速度关系恢复速度和
  加速度，再将预测位移历史送入 Steel02 纤维梁得到总恢复力 `Fint`。唯一的训练目标为
  `mean(((M*a + C*v + Fint - Fwave) / 1e5 N)^2)`。这里对完整平衡残差
  统一采用固定的 `1e5 N` 尺度，不分别缩放方程中的各项。
- E-PINN：由预测位移计算非线性恢复力并送入固定 SCL；唯一的训练目标为
  `mean((du_LSTM - du_SCL)^2)`。

以上两个 loss 都不使用标签 loss、响应全量 loss、额外权重或数据集 RMS
缩放。PINN loss 是经过固定力尺度处理的无量纲 MSE；E-PINN loss 是物理位移
增量残差的 MSE，单位为 `m^2`，因此两者的数值大小不能直接比较。

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
```

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
```
