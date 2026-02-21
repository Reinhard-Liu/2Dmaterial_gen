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
```
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#fff', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#f4f4f4'}}}%%
graph LR
    %% ==============================================================================
    %% 样式定义区域 (Style Definitions)
    %% ==============================================================================
    classDef dataContainer fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,rx:5,ry:5,color:#0d47a1,font-weight:bold;
    classDef dbContainer fill:#bbdefb,stroke:#1565c0,stroke-width:2px,shape:cyl,color:#0d47a1,font-weight:bold;
    classDef processAlgorithm fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,rx:15,ry:15,color:#e65100;
    classDef aiModel fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,rx:5,ry:5,color:#1b5e20,font-weight:bold,shape:hex;

    %% ==============================================================================
    %% 阶段 1：数据融合与伪标签生成 (Data Preparation)
    %% ==============================================================================
    subgraph Stage1 [Start: 阶段 1 - 数据融合与伪标签生成]
        direction TB
        S1_C2DB[(C2DB 原始二维\n晶体数据库)]:::dbContainer
        S1_OC20[(OC20 数据集)]:::dbContainer
        
        S1_Pretrain(DimeNet++ 预训练):::processAlgorithm
        S1_DimeNet_Pre{{DimeNet++\n(预训练模型)}}:::aiModel
        
        S1_DataClean(数据清洗 &\n提取原子特征/坐标):::processAlgorithm
        S1_Constraint2D(应用 2D-PBC 约束):::processAlgorithm
        
        S1_DimeNet_Infer{{DimeNet++\n(推理模式)}}:::aiModel
        S1_LabelCalc(标签计算转换\nΔE_H → ΔG_H):::processAlgorithm
        
        S1_PyGData[PyG 图数据集\n(含 ΔG_H, Stability, Synth 标签)]:::dataContainer

        %% Data Flow Stage 1
        S1_OC20 -- 预训练数据 --> S1_Pretrain
        S1_Pretrain -- 训练权重 --> S1_DimeNet_Pre
        S1_C2DB -- data.json, structure.json --> S1_DataClean
        S1_DataClean --> S1_Constraint2D
        S1_Constraint2D -- 约束后的晶体结构 --> S1_DimeNet_Infer
        S1_DimeNet_Pre -. 加载权重 .-> S1_DimeNet_Infer
        S1_DimeNet_Infer -- 预测表面氢吸附能 ΔE_H --> S1_LabelCalc
        S1_C2DB -- 提取 E_hull & 合成性信息 --> S1_LabelCalc
        S1_LabelCalc -- 整合所有标签 --> S1_PyGData
        S1_Constraint2D -- 整合几何特征 --> S1_PyGData
    end

    %% ==============================================================================
    %% 阶段 2：等变扩散模型训练 (Model Training)
    %% ==============================================================================
    subgraph Stage2 [阶段 2 - 等变扩散模型训练]
        direction TB
        S2_Sampler(数据采样 & 加噪过程):::processAlgorithm
        
        S2_EGNN_Backbone{{E(3)-EGNN\n核心骨干网络}}:::aiModel
        
        subgraph Heads [多任务预测头]
            direction LR
            S2_Head_Prop{{MLP头: ΔG_H}}:::aiModel
            S2_Head_Stab{{MLP头: Stability}}:::aiModel
            S2_Head_Synth{{MLP头: Synthesizability}}:::aiModel
        end
        
        S2_LossCalc(联合 Loss 计算\n(去噪 + 属性) & CFG机制):::processAlgorithm
        S2_Backprop(反向传播 & 模型更新):::processAlgorithm
        S2_TrainedModel[训练完成的\n等变扩散模型]:::dataContainer

        %% Data Flow Stage 2
        S1_PyGData ====> S2_Sampler
        S2_Sampler -- 加噪坐标 x_t, 时间步 t, 条件 c --> S2_EGNN_Backbone
        S2_EGNN_Backbone -- 潜层特征 --> Heads
        S2_EGNN_Backbone -- 预测噪声 ε_θ --> S2_LossCalc
        S2_Head_Prop -- 预测 ΔG_H --> S2_LossCalc
        S2_Head_Stab -- 预测 E_hull --> S2_LossCalc
        S2_Head_Synth -- 预测合成性 --> S2_LossCalc
        S1_PyGData -- 真实标签 Ground Truth --> S2_LossCalc
        S2_LossCalc -- 联合损失梯度 --> S2_Backprop
        S2_Backprop -- 更新权重 --> S2_EGNN_Backbone
        S2_Backprop -- 更新权重 --> Heads
        S2_EGNN_Backbone -. 最终模型状态 .-> S2_TrainedModel
        Heads -. 最终模型状态 .-> S2_TrainedModel
    end

    %% ==============================================================================
    %% 阶段 3：靶向智能生成 (Target-Driven Generation)
    %% ==============================================================================
    subgraph Stage3 [阶段 3 - 靶向智能生成 (核心创新)]
        direction TB
        S3_TemplateLib[(真实二维材料\n化学计量比模板库)]:::dbContainer
        S3_InitNoise(初始化: 采样配比 & \n高斯噪声 x_T):::processAlgorithm
        
        S3_ReverseDiff_Loop(逆向扩散循环\nStep t → t-1):::processAlgorithm
        
        S3_EGNN_Frozen{{已训练 EGNN\n(冻结参数)}}:::aiModel
        
        S3_GradientGuide(梯度制导计算\n∇|ΔG_H| + E_stab):::processAlgorithm
        S3_ConstraintEnforce(强制施加 2D 约束\nZ轴零梯度 + XY边界):::processAlgorithm
        
        S3_GeneratedCIF[生成的候选\n.cif 结构文件]:::dataContainer

        %% Data Flow Stage 3
        S2_TrainedModel ====> S3_EGNN_Frozen
        S3_TemplateLib -- 采样模板 --> S3_InitNoise
        S3_InitNoise --> S3_ReverseDiff_Loop
        
        %% The Generation Loop
        S3_ReverseDiff_Loop -- 当前噪声结构 x_t & t --> S3_EGNN_Frozen
        S3_EGNN_Frozen -- 预测去噪项 & 属性梯度 --> S3_GradientGuide
        S3_GradientGuide -- 计算目标导向梯度，修正方向 --> S3_ConstraintEnforce
        S3_ConstraintEnforce -- 物理约束后的更新结构 x_{t-1} --> S3_ReverseDiff_Loop
        
        S3_ReverseDiff_Loop -- 最终去噪结构 (t=0) --> S3_GeneratedCIF
    end

    %% ==============================================================================
    %% 阶段 4：指标评估与交付 (Evaluation & Delivery)
    %% ==============================================================================
    subgraph Stage4 [End: 阶段 4 - 指标评估与交付]
        direction TB
        S4_FullStackEval(全栈评估器\nWorkflow):::processAlgorithm
        
        S4_Eval_DimeNet{{DimeNet++\n(HER 活性评估)}}:::aiModel
        S4_Eval_MatterSim{{MatterSim MLFF\n(弛豫 & 稳定性评估)}}:::aiModel
        S4_Eval_CSLLM{{CSLLM 大语言模型\n(合成成功率评估)}}:::aiModel
        
        S4_VizReport[可视化分布图\n& 对比评估报告]:::dataContainer

        %% Data Flow Stage 4
        S3_GeneratedCIF ====> S4_FullStackEval
        S4_FullStackEval -- 1. 调用结构 --> S4_Eval_DimeNet
        S4_FullStackEval -- 2. 调用结构 --> S4_Eval_MatterSim
        S4_FullStackEval -- 3. 调用信息 --> S4_Eval_CSLLM
        
        S4_Eval_DimeNet -- 平均 ΔG_H --> S4_VizReport
        S4_Eval_MatterSim -- 稳定性分数 & 弛豫后结构 --> S4_VizReport
        S4_Eval_CSLLM -- 合成概率评分 --> S4_VizReport
    end

    %% 主要阶段之间的粗箭头连接，体现宏观流转
    S1_PyGData ==> S2_Sampler
    S2_TrainedModel ==> S3_EGNN_Frozen
    S3_GeneratedCIF ==> S4_FullStackEval

    %% 样式调整补充
    linkStyle default stroke-width:2px,fill:none,stroke:gray;
```
## 原理与公式

1. 多任务联合训练损失 (Multi-Task Training Loss)

模型训练的目标是最小化扩散重建损失与三个物理属性预测损失的加权和 ：

$$\mathcal{L}_{total}=\mathbb{E}_{t,\mathbf{x}_0,\epsilon}[\|\epsilon-\epsilon_\theta(\mathbf{x}_t,t)\|^2]+\lambda_1\|\Delta G_H^{pred}-\Delta G_H^{true}\|_1+\lambda_2\|E_{hull}^{pred}-E_{hull}^{true}\|_2^2+\lambda_3\text{BCE}(P_{synth}^{pred},y_{synth}^{true})$$

2. 梯度制导采样更新律 (Gradient-Guided Langevin Update)

在生成过程的每一步 $t \to t-1$，利用预测头计算目标属性梯度，对动力学进行引导修正：

$$\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t,t)\right)-\eta\cdot\nabla_{\mathbf{x}_t}\mathcal{L}_{target}(f_\phi(\mathbf{x}_t))+\sigma_t\mathbf{z}$$

3. HER 吉布斯自由能近似 (HER Gibbs Free Energy)

采用以下物理近似计算析氢反应的吉布斯自由能:

$$
\Delta G_{H^\ast} \approx \Delta E_{H^\ast} + \Delta E_{ZPE} - T\Delta S_{H^\ast} \approx \Delta E_{H^\ast} + 0.27 \text{ eV}
$$

其中 $\Delta E_{H^*}$ 由预训练的 DimeNet++ 模型预测。

## 实验参数表
<img width="813" height="534" alt="实验参数表" src="https://github.com/user-attachments/assets/bc3e0aa2-c4df-440f-967b-16fdfdfbc06b" />

## 评估指标

1. 平均 HER $\Delta G$ 误差 (Mean Absolute Error of $\Delta G_H$):

衡量生成材料催化活性与理想值 (0 eV) 的平均偏差 。

$$\text{MAE}_{\Delta G}=\frac{1}{N}\sum_{i=1}^{N}|\Delta G_{H, i}^{pred}-0|$$

2. 稳定性得分 (Stability Score):

基于机器学习力场 (MatterSim) 预测的形成能 $E_{form}$ 计算的归一化得分 。

$$\text{Score}_{stab}=\frac{1}{N}\sum_{i=1}^{N}\exp(-\max(0,E_{form, i}-E_{stable}^{ref}))$$

3. 合成成功率 (Synthesis Success Rate):

基于材料大模型 (CSLLM) 预测判定为“可合成”的材料占比 。

$$\text{Rate}_{synth}=\frac{\text{Count}(\text{Predicted as Synthesizable})}{N}\times100\%$$

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
