# Euler_Beam

## 1. 问题定义
Euler Beam 公式

$$ \frac{\partial ^4 u}{\partial x^4} + 1 = 0, x \in [0,1]$$

边界条件：

$$ u^{''}(1)=0, u^{'''}(1)=0 $$

狄利克雷条件：

$$ u(0)=0 $$

诺依曼边界条件：

$$ u^{'}(0)=0 $$

## 2. 项目依赖
- Python 3.7
- Mindspore = 2.3.0-rc1
- MindTorch = 0.3.0

## 3. MindTorch相关
### 3.1 简介
MindTorch是一个旨在将PyTorch训练脚本无缝迁移到MindSpore框架以执行的工具，其设计目标是在保持PyTorch用户原有使用习惯不变的同时，确保PyTorch代码在异腾（或其他支持平台）上获得卓越的性能。

![MindTorch示意图](https://github.com/KairosXu/Euler_Beam/blob/main/asserts/picture.png)

目前MindTorch主要有PyTorch和TorchVision两个接口，其功能具体分别如下：
- PyTorch接口支持：MindTorch目前支持大部分PyTorch常用接口适配。用户接口使用方式不变，基于MindSpore动态图或静态图模式下执行在异腾算力平台上。可以在torch接口支持列表中查看接口支持情况。
- TorchVision接口支持：MindTorch TorchVision是迁移自PyTorch官方实现的计算机视觉工具库，延用PyTorch官方API设计与使用习惯，内部计算调用MindSpore算子，实现与torchvision原始库同等功能。可以在TorchVision接口支持列表中查看接口支持情况。

### 3.2 安装
通过pip安装

```
pip install mindtorch (MindSpore版本 >= 2.2.1)
```

通过源码安装

```
git clone https://git.openi.org.cn/OpenI/MSAdapter.git
cd MSAdapter
python setup.py install
```

### 3.3 使用MindTorch
由于我们安装的是最新版本的MindTorch，在代码文件主入口导入mstorch_enable包即可将PyTorch代码适配到MindTorch，从而在NPU设备上运行PyTorch代码。

```
from mindtorch.tools import mstorch_enable
```

导入mstorch_enable后，代码执行时torch同名的导入模块会自动被转换为mindtorch相应的模块（目前支持torch、torchvision、torchaudio相关模块的自动转换），接下来执行主入口的.py文件即可。MindTorch其他版本的使用方式可以参考[使用指南](https://mindtorch.readthedocs.io/zh-cn/latest/docs/User_Guide_Import.html)。

## 4. 问题求解
### 4.1 模型构建
在 Euler Beam 问题中，每一个已知的坐标点 $x$ 都有对应的待求解的未知量 $u$ ，我们在这里使用一个简单的多层感知机 $MLP$ 来表示 $x$ 到 $u$ 的映射函数 $f:R^1 \rightarrow R^1$ ，即：

$$
u = f(x)
$$

上式中 $f$ 即为 $MLP$ 模型本身，代码表示如下：

```
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
```

### 4.2 计算域构建
Euler Beam 问题作用在 $[0,1]$ 的一维区域上:

```
t_physics = torch.linspace(0, 1, 1000).view(-1, 1).requires_grad_(True)
```

同时对于Euler Beam问题还需要定义边界点，用于边界损失计算：

```
t_boundary_0 = torch.tensor(0.).view(-1, 1).requires_grad_(True)
t_boundary_1 = torch.tensor(1.).view(-1, 1).requires_grad_(True)
```

### 4.3 方程与约束构建
首先需要计算计算域上的点的输出值关于输入值的四阶导数：

```
u = eb_model(t_physics)  # 使用神经网络计算物理点的输出
dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]  # 计算物理点输出的时间导数
d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0] # 二阶导数
d3udt3 = torch.autograd.grad(d2udt2, t_physics, torch.ones_like(d2udt2), create_graph=True)[0] # 三阶导数
d4udt4 = torch.autograd.grad(d3udt3, t_physics, torch.ones_like(d3udt3), create_graph=True)[0] # 四阶导数
```

接着使用损失对Euler Beam 方程进行约束：

```
loss4 = torch.mean((d4udt4 + 1) ** 2)  # 计算方程损失
```

边界条件约束：

```
u1 = model(t_boundary_1)
du1dt = torch.autograd.grad(u1, t_boundary_1, torch.ones_like(u1), create_graph=True)[0]
d2u1dt2 = torch.autograd.grad(du1dt, t_boundary_1, torch.ones_like(du1dt), create_graph=True)[0]
d3u1dt3 = torch.autograd.grad(d2u1dt2, t_boundary_1, torch.ones_like(d2u1dt2), create_graph=True)[0]
loss2 = (torch.squeeze(d2u1dt2) - 0) ** 2
loss3 = (torch.squeeze(d3u1dt3) - 0) ** 2
```

狄利克雷条件约束：

```
u0 = pinn(t_boundary_0)
loss0 = (torch.squeeze(u0) - 0) ** 2
```

诺依曼边界条件约束：

```
du0dt = torch.autograd.grad(u0, t_boundary_0, torch.ones_like(u0), create_graph=True)[0]
loss1 = (torch.squeeze(du0dt) - 0) ** 2
```

### 4.4 优化器构建
训练过程会调用优化器来更新模型参数，此处选择较为常用的 Adam 优化器：

```
optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)
```

### 4.5 可视化
在模型评估时，我们使用matplotlib中的pyplot对模型 $f$ 的预测结果进行可视化，并保存成png格式的图像。

```
u_pred = eb_model(x_test).detach()
plt.figure(figsize=(6,2.5))
plt.plot(x_test, u_pred, label="u_pred")
plt.plot(x_test, u_exact, label="u_label")
plt.legend()
plt.savefig("result.png")
```

## 5. 训练脚本
使用以下命令运行程序即可：

```
git clone https://github.com/KairosXu/Euler_Beam.git
cd Euler_Beam
python eluer_beam.py
```

相关的参数可以在[配置文件](https://github.com/KairosXu/Euler_Beam/blob/main/configs/default.py)中进行修改，包括训练轮次、随机种子、优化器学习率等等。

## 6. 结果展示
在[0,1]区间内线性采样若干个点 $x$ ,使用训练好的模型 $f$ 预测其求解量 $u \\_ pred$，并与其真实值进行对比，结果如下：
![](https://github.com/KairosXu/Euler_Beam/blob/main/asserts/result.png)
