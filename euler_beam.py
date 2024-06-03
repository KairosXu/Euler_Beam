import torch
import argparse
import os.path as osp
import matplotlib.pyplot as plt

from models.fcn import FCN
from train import train_euler_beam
from configs.default import get_config
from tools.utils import save_checkpoint, set_seed, get_logger


def parse_option():
    parser = argparse.ArgumentParser(description='Euler Beam Problem')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def exact_solution(x):
    u = -(x**4) / 24 + x**3 / 6 - x**2 / 4
    return u


def main(config, logger):
    eb_model = FCN(1, 1, 32, 3)
    # 定义边界点，用于边界损失计算
    t_boundary_0 = torch.tensor(0.).view(-1, 1).requires_grad_(True)  # 创建一个单元素张量，值为0，形状为(1, 1)，需要计算梯度
    t_boundary_1 = torch.tensor(1.).view(-1, 1).requires_grad_(True)  # 创建一个单元素张量，值为1，形状为(1, 1)，需要计算梯度

    # 定义域上的训练点，用于物理损失计算
    t_physics = torch.linspace(0, 1, 1000).view(-1, 1).requires_grad_(True)  # 创建一个从0到1等间隔的30个点的张量，形状为(1000, 1)，需要计算梯度

    # 训练过程
    x_test = torch.linspace(0, 1, 300).view(-1, 1)  # 创建一个测试点集，用于最后的可视化
    u_exact = exact_solution(x_test)  # 计算精确解，用于与PINN解进行对比
    optimiser = torch.optim.Adam(eb_model.parameters(), lr=config.TRAIN.OPTIMIZER.LR)  # 使用Adam优化器
    train_euler_beam(config, eb_model, optimiser, t_boundary_0, t_boundary_1, t_physics, logger)

    # 使用训练好的神经网络 pinn 对测试点 t_test 进行预测，并使用 .detach() 从当前计算图中分离，便于后续处理
    u_pred = eb_model(x_test).detach()
    plt.figure(figsize=(6,2.5))
    plt.plot(x_test, u_pred, label="u_pred")
    plt.plot(x_test, u_exact, label="u_label")
    plt.legend()
    plt.savefig(config.OUTPUT + "/result.png")
    

if __name__ == '__main__':
    config = parse_option()

    # Set random seed
    set_seed(config.SEED)
    # get logger
    if not config.EVAL_MODE:
        output_file = osp.join(config.OUTPUT, 'log_train_.log')
    else:
        output_file = osp.join(config.OUTPUT, 'log_test.log')
    logger = get_logger(output_file, 0, 'euler_beam')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")
    main(config, logger)