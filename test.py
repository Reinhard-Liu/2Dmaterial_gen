import os
import torch
from models.diffusion_model import DenoisingEGNN
from models.structure_generator import StructureGenerator
from utils.geo_utils import MetricsEvaluator
from utils.vis import Visualizer


def main():
    print("🚀 启动阶段 5：模型推理与 Baseline 指标对齐交付")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载最佳模型权重
    print("加载训练好的最佳扩散模型权重...")
    model = DenoisingEGNN(num_node_features=1, hidden_dim=128, num_layers=4)
    checkpoint_path = 'results/checkpoints/best_diffusion_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ 成功加载权重: {checkpoint_path}")
    else:
        print("⚠️ 未找到 Checkpoint，将使用随机权重进行流程演示。")

    # 2. 批量生成靶向材料
    generator = StructureGenerator(model=model, device=device, num_steps=200)
    gen_z, gen_pos, gen_batch = generator.generate_guided_2d_materials(
        num_materials=20, # 你可以根据需要改回 100
        num_atoms_per_mat=12,
        guidance_scale=0.08,
        target_delta_g=0.0
    )

    # 获取 ASE Atoms 列表并导出 CIF（满足“保存在results的子文件夹中”的要求）
    cif_output_dir = 'results/generated_cifs'
    atoms_list = generator.export_to_atoms_and_cif(gen_z, gen_pos, gen_batch, output_dir=cif_output_dir)
    print(f"💾 {len(atoms_list)} 个晶体结构 .cif 文件已存入: {cif_output_dir}")

    # 3. Baseline 指标计算
    evaluator = MetricsEvaluator(
        dimenet_weights_path=r'D:\Programming Software\github_project\MachineLearning_MG\models\weights\dimenet_best_ocp.pth'
    )
    results_df = evaluator.run_full_evaluation(atoms_list)

    # 保存表格数据
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/generation_metrics_report.csv', index=False)

    # 4. 效果可视化保存（直接传入 'results' 即可存放三个主图）
    vis = Visualizer(output_dir='results')
    vis.generate_all_plots(results_df, atoms_list)

    print("请在 results/ 文件夹下查看交付结果：")
    print(" ├─ loss_curve.png (通过 train.py 生成)")
    print(" ├─ her_performance.png")
    print(" ├─ stability_curve.png")
    print(" ├─ generated_structures.png")
    print(" └─ generated_cifs/ (包含所有具体的结构文件)")

if __name__ == "__main__":
    main()