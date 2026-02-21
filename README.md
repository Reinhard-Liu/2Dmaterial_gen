# Intelligent 2D Material Generation via Equivariant Diffusion Models

本项目旨在利用等变扩散模型（Equivariant Diffusion Models）和智能梯度靶向优化手段，从通用晶体数据库中学习材料结构特征，逆向设计并生成具有高 HER（析氢反应）催化活性、良好热力学/动力学稳定性以及较强实验可合成性的新型二维材料 。

本项目基于现有baseline重新设计并优化得出，并与baseline设计的材料进行了对比。

baseline链接：https://github.com/deamean/material_generation?tab=readme-ov-file#2-materials-project-api%E5%AF%86%E9%92%A5

## 项目结构
```mermaid
flowchart TD
    %% ================= 样式定义 (Material Design 色系) =================
    classDef data fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1,rx:8px,ry:8px;
    classDef process fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#E65100,rx:8px,ry:8px;
    classDef model fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20,rx:8px,ry:8px;
    classDef output fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#4A148C,rx:8px,ry:8px;

    %% ================= 阶段一：数据准备 =================
    subgraph Phase1 ["📊 阶段一：数据准备 (Data Preparation)"]
        direction TB
        A["C2DB 原始数据<br/>(data.json / structure.json)"]:::data --> B("数据清洗与 2D 约束处理<br/>(material_dataset.py)"):::process
        B --> C{"DimeNet++ 伪标签生成<br/>(quick_formation_screening.py)"}:::process
        C -- "预测 ΔE_H" --> D["带标签的 PyG 图数据集<br/>(processed/*.pt)"]:::data
    end

    %% ================= 阶段二：模型训练 =================
    subgraph Phase2 ["🧠 阶段二：模型训练 (Training Phase)"]
        direction TB
        D --> E("E3-EGNN 扩散模型骨干<br/>(diffusion_model.py)"):::model
        D --> F("多任务属性预测头<br/>(HER / Stability / Synth)"):::model
        E --> G{"联合损失函数计算<br/>(optimization.py)"}:::process
        F --> G
        G -. "反向传播优化" .-> E
        G -. "反向传播优化" .-> F
    end

    %% ================= 阶段三：靶向生成 =================
    subgraph Phase3 ["🎯 阶段三：靶向生成 (Target-Driven Generation)"]
        direction TB
        H["高斯噪声 x_T"]:::data --> I("结构生成器<br/>(structure_generator.py)"):::process
        I -- "1. EGNN 去噪预测" --> J["中间状态 x_t"]:::data
        
        %% 独立推理节点，避免跨子图连线导致画面杂乱
        J -- "2. 计算属性梯度 ∇L" --> F_infer("调用多任务属性预测头"):::model
        F_infer -- "3. 梯度回传指导修正" --> I
        
        I -. "4. 更新坐标并循环 T 步" .-> J
        J ===> K["最终生成的 2D 结构<br/>(.cif files)"]:::output
    end

    %% ================= 阶段四：评估与可视化 =================
    subgraph Phase4 ["📈 阶段四：评估与可视化 (Evaluation)"]
        direction TB
        K --> L("全栈评估器<br/>(geo_utils.py / test.py)"):::process
        L -- "MatterSim / CSLLM / DimeNet" --> M["可视化图表与指标报告<br/>(results/*.png)"]:::output
    end

    %% ================= 跨阶段层级约束 (保持整体从上到下的整洁排版) =================
    Phase1 ~~~ Phase2
    Phase2 ~~~ Phase3
    Phase3 ~~~ Phase4
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
