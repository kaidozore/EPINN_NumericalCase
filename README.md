# 5 自由度纤维梁 PINN / E-PINN

本工程读取 `E:\MDOF_case` 中 MATLAB 已生成的 5 自由度数据。Python 不重新
组装 `M、C、K0`，也不生成波浪荷载。

## MATLAB 输入

- `wave_loads_300.mat`：读取 `Fwave、time、M、C、K0`，并从 `cfg` 读取缩聚
  变换、单元自由度、纤维划分、Gauss 积分点及 Steel02 参数。
- `wave_responses_newmark_300.mat`：读取监督标签 `U、V、Acc` 和校核量
  `FintFull`。
- `wave_responses_etdm_300.mat`：读取 `ETDM/A`、`Fnonlinear` 以及
  ETDM 参考响应。

h5py 读取后的时程统一为 `[sample, time, 5]`。当前正式 MATLAB 文件需要重新
生成 300 个样本后再进行论文训练；目录中已有的 5 个样本可以用于接口检查。
不足 200 个样本的调试数据会保留最后 1 个样本作为独立测试集；300 样本正式
数据仍严格采用 170/30/100 划分。

## 模型

两种模型均采用三层 LSTM，LSTM 层之间没有 ReLU；其后为
`FC -> ReLU -> FC`。

- PINN 预测 5 个平移自由度的位移增量，累加得到位移，并通过可微 Steel02
  纤维梁模块计算 `Fint` 与 `g=Fint-K0*u`。物理残差为
  `M*a+C*v+K0*u+g-P`。
- E-PINN 预测 5 个位移增量，通过相同的纤维梁模块计算 5 维 `g`，再把
  `[P,g]` 输入由 MATLAB `ETDM/A` 固定构造的 SCL。MATLAB 已在
  `LF=[I,-I]` 中包含负号，因此 Python 不重复改变 `g` 的符号。

MATLAB 强制保存 `Fwave(:,1,:)=0`，且初始状态和 `Fnonlinear(:,1,:)` 均为零，
因此 SCL 只加载 `ETDM/A` 作为固定权重，不读取 `T、Q1、Q2`。

## 本地静态检查

静态检查不会执行优化器更新或正式训练：

```powershell
python static_check.py --data-root E:\MDOF_case
```

检查内容包括 MATLAB 字段与维度、Steel02 完整时程恢复力、Aij/SCL 排列与
符号、PINN 前向/损失/反向和 E-PINN 前向/损失/反向。

## AutoDL 训练

在 AutoDL 中上传整个 `MDOF_case`（或使用 `--data-root` 指向三个 MATLAB
文件所在目录），安装依赖：

```bash
pip install -r python/requirements.txt
```

`ninja` 是编译 Steel02 自定义 C++/CUDA 扩展的必需依赖，已经列入
`requirements.txt`。可单独检查：

```bash
python -c "import ninja; print(ninja.__version__)"
ninja --version
```

首次在 CUDA 设备上运行时会编译扩展，成功后终端显示
`Steel02 CUDA extension loaded successfully.`；后续运行复用
`extensions/_build_cuda/` 中的构建缓存。
对于计算能力 8.9 的 Ada 显卡，如果系统 NVCC 低于 CUDA 11.8，加载器会自动
改用 `8.6+PTX` 前向兼容编译，避免 `Unsupported gpu architecture
'compute_89'`。

先用短序列确认 CUDA 环境：

```bash
cd python
python static_check.py --data-root .. --check-steps 32 --constitutive-steps 200
python EPINN_MDOFSys_Train.py --data-root .. --epochs 1 --batch-size 2 --sequence-length 64 --time-truncation 64
python PINN_MDOFSys_Train.py --data-root .. --epochs 1 --batch-size 2 --sequence-length 64
```

短测试通过后，去掉 `--sequence-length` 开展完整时程训练：

```bash
python EPINN_MDOFSys_Train.py --data-root ..
python PINN_MDOFSys_Train.py --data-root ..
```

PINN采用阶段化训练。前`--physics-warmup-epochs`轮只遍历40个有标签样本，
完全跳过Steel02和物理残差；训练与验证均记录同一定义的监督loss。warmup结束
后切换到170个训练样本，并在`--physics-ramp-epochs`轮内平方渐增物理权重。
两个阶段的最优模型分别保存在`checkpoints/warmup/`和
`checkpoints/physics/`，避免不同loss定义的checkpoint互相竞争。

每个 epoch 的 `train_loss` 和 `val_loss` 写入当前时间戳目录下的
`epoch_loss.csv`，同时覆盖更新 `epoch_loss.png`。程序每个 epoch 保存一个候选
检查点，再按照 `val_loss` 排序，在本次训练的 `checkpoints` 文件夹中只保留
最小的 10 个 `.pth`；其他历史时间戳训练目录不会自动删除。

预测脚本读取训练生成的 `.pth` 文件：

```bash
python EPINN_MDOFSys_Predict.py checkpoint.pth --data-root ..
python PINN_MDOFSys_Predict.py checkpoint.pth --data-root ..
```
