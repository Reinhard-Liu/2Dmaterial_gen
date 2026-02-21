# Intelligent 2D Material Generation via Equivariant Diffusion Models

本项目旨在利用等变扩散模型（Equivariant Diffusion Models）和智能梯度靶向优化手段，从通用晶体数据库中学习材料结构特征，逆向设计并生成具有高 HER（析氢反应）催化活性、良好热力学/动力学稳定性以及较强实验可合成性的新型二维材料 。

本项目基于现有baseline重新设计并优化得出，并与baseline设计的材料进行了对比。

baseline链接：https://github.com/deamean/material_generation?tab=readme-ov-file#2-materials-project-api%E5%AF%86%E9%92%A5

## 项目结构

```
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
```

## 模型结构

## 原理与公式

1. 多任务联合训练损失 (Multi-Task Training Loss)

模型训练的目标是最小化扩散重建损失与三个物理属性预测损失的加权和 ：

$$\mathcal{L}_{total}=\mathbb{E}_{t,\mathbf{x}_0,\epsilon}[\|\epsilon-\epsilon_\theta(\mathbf{x}_t,t)\|^2]+\lambda_1\|\Delta G_H^{pred}-\Delta G_H^{true}\|_1+\lambda_2\|E_{hull}^{pred}-E_{hull}^{true}\|_2^2+\lambda_3\text{BCE}(P_{synth}^{pred},y_{synth}^{true})$$

2. 梯度制导采样更新律 (Gradient-Guided Langevin Update)

在生成过程的每一步 $t \to t-1$，利用预测头计算目标属性梯度，对动力学进行引导修正：

$$\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t,t)\right)-\eta\cdot\nabla_{\mathbf{x}_t}\mathcal{L}_{target}(f_\phi(\mathbf{x}_t))+\sigma_t\mathbf{z}$$

3. HER 吉布斯自由能近似 (HER Gibbs Free Energy)

采用以下物理近似计算析氢反应的吉布斯自由能 ：

$$\Delta G_{H^*}\approx\Delta E_{H^*}+\Delta E_{ZPE}-T\Delta S_{H^*}\approx\Delta E_{H^*}+0.27\text{ eV}$$

## 实验参数表

Parameter Category,Parameter Name,Value / Setting,Description
Data & Features,Dataset Source,C2DB (filtered),"2D materials only, < 60 atoms"
,Data Augmentation,"Rotation, Strain (±2%)",Essential for small datasets
Model Architecture,Backbone,E(3)-EGNN,Equivariant Graph Neural Network
,Hidden Dimension,128,Dimension of node/edge embeddings
Training,Diffusion Steps (T),1000,Total timesteps for noise schedule
,Optimizer,AdamW,"Weight decay = 1e-4, LR = 1e-4"
,CFG Probability,0.15,Prob. of dropping conditions
,Loss Weights (λ),"λ1​=1.0,λ2​=1.0,λ3​=0.5","Weights for HER, Stab, Synth"
Generation,Guidance Scale (η),0.05 - 0.10,Strength of gradient correction

## 创新点说明

本项目在底层算法和物理约束上进行了深度创新：

1. 使用了基于扩散模型的材料生成框架，并结合智能优化手段提升HER催化活性和稳定性 。

2. 生成即优化 (Gradient-Guided Generation)：不依赖于海量随机生成后的事后筛选，而是在扩散降噪的每一步中显式注入性能优化的物理梯度引导，使得高活性、高稳定性材料的生成命中率实现了指数级跃升。

3. 模板驱动与无悬挂键边界设计：通过内置真实的二维晶体配方模板，配合紧凑的 XY 周期性边界构建（Compact PBC），从根本上解决了生成二维材料时易出现结构破碎和悬挂键的问题。

4. E(3) 等变性物理先验：使用 EGNN 替代普通 GNN，用数学上的对称性弥补了数据量的不足，极大提升了模型从有限数据库中学习通用结构特征的效率。

## 结果整体可视化分析
### 1. ΔG_H性能图
![ΔG_H性能图](./results/her_performance.png)
### 2. 稳定性与合成性评估曲线
![稳定性与合成性评估曲线](./results/stability_curve.png)
### 3. 生成的材料结构图
![生成的材料结构图](./results/generated_structures.png)

## 与baseline的对比（通过MatterSim、CSLLM、DimeNet++统一评定三项指标）
baseline生成的材料保存在results_external文件夹中，评估结果通过evaluate_external.py给出。我的评估结果在运行test.py会自动给出。虽然是通过两个代码实现，但是采用的方法和模型均为一致。
<img width="554" height="389" alt="数据对比表" src="https://github.com/user-attachments/assets/d3164405-1173-4952-8e9c-4aa9feac9788" />
