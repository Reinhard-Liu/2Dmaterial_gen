# Intelligent 2D Material Generation via Equivariant Diffusion Models

本项目旨在利用等变扩散模型（Equivariant Diffusion Models）和智能梯度靶向优化手段，从通用晶体数据库中学习材料结构特征，逆向设计并生成具有高 HER（析氢反应）催化活性、良好热力学/动力学稳定性以及较强实验可合成性的新型二维材料 。

## 项目结构

'''
project/
├── models/
│   ├── diffusion_model.py      # E(3)-等变扩散网络与多任务预测头
│   ├── structure_generator.py  # 基于真实模板与梯度引导的采样器
│   ├── dimnet_model.py         # DimeNet++ 表面吸附能预测模型 (HER伪标签)
│   └── optimization.py         # 多任务联合损失函数与学习率调度器
├── dataset/
│   └── material_dataset.py     # C2DB 图数据集构建与 2D-PBC 约束处理
├── train.py                    # 模型多任务联合训练主入口
├── test.py                     # 靶向生成、评估流水线与交付主入口
├── evaluate_external.py        # 外部材料的CIF批量评估脚本
├── pretrain_dimenet.py         # DimeNet++ 预训练脚本
├── utils/
│   ├── geo_utils.py            # 材料稳定性计算、HER性能评估
│   └── vis.py                  # 结果可视化引擎
├── README.md                   # 项目说明文档
└── results/                    # 运行生成的图表与结构输出目录
'''
