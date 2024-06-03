import torch
from tqdm import tqdm

def train_euler_beam(config, model, optimiser, t_boundary_0, t_boundary_1, t_physics, logger):
    model.train()
    for i in tqdm(range(config.TRAIN.MAX_EPOCH)):
        optimiser.zero_grad()  # 在每次迭代开始时清空梯度

        # 计算每项损失
        lambda0, lambda1, lambda2, lambda3, lambda4 = 1e-1, 1e-1, 1e-1, 1e-1, 1e-3  # 设置损失函数中的超参数

        # 计算边界损失
        u0 = model(t_boundary_0)  # 使用神经网络计算边界点0的输出
        u1 = model(t_boundary_1)  # 使用神经网络计算边界点1的输出

        # 0和1阶导损失
        loss0 = (torch.squeeze(u0) - 0) ** 2  # 计算边界损失的第一部分
        du0dt = torch.autograd.grad(u0, t_boundary_0, torch.ones_like(u0), create_graph=True)[0]  # 计算边界点输出的时间导数
        loss1 = (torch.squeeze(du0dt) - 0) ** 2  # 计算边界损失的第二部分

        # 2和3阶导损失
        du1dt = torch.autograd.grad(u1, t_boundary_1, torch.ones_like(u1), create_graph=True)[0]
        d2u1dt2 = torch.autograd.grad(du1dt, t_boundary_1, torch.ones_like(du1dt), create_graph=True)[0]
        d3u1dt3 = torch.autograd.grad(d2u1dt2, t_boundary_1, torch.ones_like(d2u1dt2), create_graph=True)[0]
        loss2 = (torch.squeeze(d2u1dt2) - 0) ** 2
        loss3 = (torch.squeeze(d3u1dt3) - 0) ** 2

        # 计算物理损失
        u = model(t_physics)  # 使用神经网络计算物理点的输出
        dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]  # 计算物理点输出的时间导数
        d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]  # 计算物理点输出的二阶时间导数
        d3udt3 = torch.autograd.grad(d2udt2, t_physics, torch.ones_like(d2udt2), create_graph=True)[0]
        d4udt4 = torch.autograd.grad(d3udt3, t_physics, torch.ones_like(d3udt3), create_graph=True)[0]
        loss4 = torch.mean((d4udt4 + 1) ** 2)  # 计算物理损失

        # 反向传播并更新参数
        loss = lambda0 * loss0 + lambda1 * loss1 + lambda2 * loss2 + lambda3 * loss3 + lambda4 * loss4# 计算总损失
        if i % 10 == 0:
            logger.info("epoch:{}, loss:{}".format(i, loss))
        loss.backward()  # 反向传播
        optimiser.step()  # 更新网络参数