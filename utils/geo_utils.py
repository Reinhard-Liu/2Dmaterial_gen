import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# ASE 与 Pymatgen 工具
from ase.optimize import BFGS
from ase.neighborlist import neighbor_list
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data

# ==========================================
# 1. 尝试导入 Baseline 所需的真实评估依赖
# ==========================================

# A. 微软 MatterGen / MatterSim
try:
    from mattersim.forcefield import MatterSimCalculator

    MATTERSIM_AVAILABLE = True
except ImportError:
    MATTERSIM_AVAILABLE = False
    print("⚠️ 未检测到 mattersim。请配置 Microsoft MatterSim 环境以启用 ML 力场弛豫。")

# B. CSLLM (基于 Transformers 的大语言模型)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ 未检测到 transformers。请运行 `pip install transformers` 以启用 CSLLM 评估。")

# C. 引入我们自己训练的 DimeNet++ 构建函数 (用于表面吸附能预测)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dimnet_model import build_dimenet_plus_plus


# ==========================================
# 2. 综合指标验证器
# ==========================================
class MetricsEvaluator:
    def __init__(self, device='cuda', dimenet_weights_path='results/checkpoints/dimenet_best_ocp.pth'):
        print("🔧 初始化全栈真实指标验证器 (MatterSim / Adsorption ML / CSLLM)...")
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 1. 初始化吸附能评估器
        print("  ├─ 加载吸附能预测模型 (DimeNet++)...")
        self.her_model = self._load_dimenet(dimenet_weights_path)

        # 2. 初始化 MatterSim ML 力场计算器
        if MATTERSIM_AVAILABLE:
            print("  ├─ 加载 Microsoft MatterSim 预训练势场...")
            # 默认加载 MatterSim 的通用预训练模型 (根据其实际 API 可调节 load_path)
            self.mattersim_calc = MatterSimCalculator(device=self.device)
        else:
            self.mattersim_calc = None

        # 3. 初始化 CSLLM
        if TRANSFORMERS_AVAILABLE:
            print("  └─ 加载 CSLLM 可合成性大模型...")
            # 本地 CSLLM 模型权重路径
            self.llm_model_name = r"D:\Programming Software\github_project\MachineLearning_MG\models\csllm\llama3-8bf-hf"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
                self.csllm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                ).to(self.device)
                self.csllm_model.eval()
            except Exception as e:
                self.csllm_model = None
                print(f"  └─ CSLLM 权重加载失败 (请确保模型已下载): {e}")
        else:
            self.csllm_model = None

    def _load_dimenet(self, weights_path):
        """加载训练好的吸附能预测模型权重"""
        if not os.path.exists(weights_path):
            print(f"⚠️ 找不到吸附能模型权重: {weights_path}。将使用未训练的初始化权重。")
        model = build_dimenet_plus_plus(self.device)
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        model.eval()
        return model

    # ==========================================
    # 核心评估方法 1：吸附能预测 (表面催化稳定性)
    # ==========================================
    def evaluate_delta_g(self, atoms):
        """计算 HER 活性 (ΔG_H)"""
        # 将 ASE Atoms 转换为 PyG 图结构
        atoms.pbc = [True, True, False]
        edge_i, edge_j, cell_offsets = neighbor_list('ijS', atoms, 5.0)

        z = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float)
        edge_index = torch.vstack([torch.tensor(edge_i), torch.tensor(edge_j)]).long()

        with torch.no_grad():
            z_device = z.to(self.device)
            pos_device = pos.to(self.device)
            batch = torch.zeros(z.shape[0], dtype=torch.long).to(self.device)

            # 预测吸附能并加上 0.27 eV 的零点能与熵变修正
            delta_e_h = self.her_model(z_device, pos_device, batch)
            delta_g_h = delta_e_h.item() + 0.27

        # 依据火山图原理，取绝对值评价
        return delta_g_h, abs(delta_g_h)

    # ==========================================
    # 核心评估方法 2：MatterSim 热力学稳定性
    # ==========================================
    def evaluate_mattersim_stability(self, atoms):
        """使用 MatterSim 对结构进行弛豫并计算形成能分数"""
        if self.mattersim_calc is None:
            return 0.0, 0.0

        atoms_copy = atoms.copy()
        atoms_copy.calc = self.mattersim_calc

        # 1. 结构弛豫：让 ML 力场寻找局部能量最低点
        try:
            opt = BFGS(atoms_copy, logfile=None)
            opt.run(fmax=0.05, steps=50)

            # 2. 获取弛豫后的总能量 (eV)
            total_energy = atoms_copy.get_potential_energy()
            num_atoms = len(atoms_copy)
            e_per_atom = total_energy / num_atoms

            # (注: 严谨的形成能需减去单质参考态能量。在此采用基线中的归一化平移逻辑)
            e_form = e_per_atom

            # 3. 打分逻辑：映射到 0-1 区间
            score = max(0.0, 1.0 - np.exp(e_form + 0.5))
            return float(e_form), float(score)
        except Exception as e:
            print(f"MatterSim 弛豫失败: {e}")
            return 0.0, 0.0

    # ==========================================
    # 核心评估方法 3：CSLLM 合成可行性预测
    # ==========================================
    def evaluate_csllm_synthesis(self, atoms):
        """将 CIF 序列化后送入 CSLLM 预测合成率"""
        if self.csllm_model is None:
            return 0.0

        # 将结构转化为 CIF 文本
        structure = AseAtomsAdaptor.get_structure(atoms)
        cif_string = structure.to(fmt="cif")

        # 按照 CSLLM 的预训练指令格式构造 Prompt
        prompt = (
            "Determine whether the following material can be successfully synthesized in experiments "
            "based on its CIF structure. Provide a probability score between 0.0 and 1.0.\n\n"
            f"{cif_string}\n\nProbability:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)

        try:
            with torch.no_grad():
                #
                outputs = self.csllm_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    return_dict_in_generate=True
                )

            generated_text = self.tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:],
                                                   skip_special_tokens=True)

            # 清洗并提取浮点数
            clean_text = ''.join(c for c in generated_text if c.isdigit() or c == '.')
            synth_prob = float(clean_text)

            # 确保在 0-1 之间
            return min(max(synth_prob, 0.0), 1.0)
        except Exception as e:
            return 0.0

    # ==========================================
    # 执行全流程并输出汇总报告
    # ==========================================
    def run_full_evaluation(self, atoms_list):
        print(f"\n📊 开始对 {len(atoms_list)} 个材料执行深度评估...")
        results = []

        for i, atoms in enumerate(tqdm(atoms_list, desc="Evaluating with Baselines")):
            # 1. 表面催化活性评估
            delta_g, abs_delta_g = self.evaluate_delta_g(atoms)

            # 2. MatterSim 热力学弛豫
            e_form, stab_score = self.evaluate_mattersim_stability(atoms)

            # 3. CSLLM 合成可行性推断
            synth_prob = self.evaluate_csllm_synthesis(atoms)

            # 基线综合评分公式 (参照 quick_formation_screening 权重)
            # 假设权重：HER 0.4, 稳定性 0.3, 合成率 0.3
            her_score = max(0.0, 1.0 - abs_delta_g * 2.0)  # 越靠近 0 分数越高
            composite_score = 0.4 * her_score + 0.3 * stab_score + 0.3 * synth_prob

            results.append({
                "Material_ID": i + 1,
                "Delta_G_H (eV)": delta_g,
                "Abs_Delta_G_H (eV)": abs_delta_g,
                "Formation_Energy (eV/atom)": e_form,
                "Stability_Score": stab_score,
                "Synthesis_Prob": synth_prob,
                "Is_Synthesizable": int(synth_prob > 0.5),
                "Composite_Score": composite_score
            })

        df = pd.DataFrame(results)

        print("\n" + "=" * 55)
        print("🏆 最终交付指标总结报告 (Baseline 级真实测算) 🏆".center(45))
        print("=" * 55)
        print(f"🔹 1. 平均 HER 活性 (MAE) : {df['Abs_Delta_G_H (eV)'].mean():.4f} eV")
        print(f"🔹 2. MatterSim 稳定性分数: {df['Stability_Score'].mean():.4f} / 1.0")
        print(f"🔹 3. CSLLM 可合成成功率  : {(df['Is_Synthesizable'].mean() * 100):.1f} %")
        print(f"🌟 综合评级得分 (Composite): {df['Composite_Score'].mean():.4f} / 1.0")
        print("=" * 55)

        return df