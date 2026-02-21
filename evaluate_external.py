import os
import glob
from ase.io import read
from utils.geo_utils import MetricsEvaluator
from utils.vis import Visualizer


def main():
    # 1. 设定输入和输出的文件夹路径
    input_dir = "external_cifs"  # 你存放 52 个 CIF 文件的文件夹
    output_dir = "results_external"  # 评估结果独立存放，以免覆盖你之前跑出的 results

    print(f"🚀 启动外部材料批量评估流水线...")

    # 获取所有的 .cif 文件路径
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))
    if not cif_files:
        print(f"❌ 在 '{input_dir}' 目录下没有找到任何 .cif 文件！请确认文件存放位置。")
        return

    print(f"📂 找到 {len(cif_files)} 个 CIF 文件，正在加载结构...")

    # 2. 将所有的 CIF 转化为 ASE 的 Atoms 对象，供评估器使用
    atoms_list = []
    valid_cif_names = []
    for cif_path in cif_files:
        try:
            atoms = read(cif_path)
            atoms_list.append(atoms)
            valid_cif_names.append(os.path.basename(cif_path))
        except Exception as e:
            print(f"⚠️ 读取文件 {cif_path} 失败，已跳过。报错信息: {e}")

    if not atoms_list:
        print("❌ 没有成功加载任何结构，程序退出。")
        return

    # 3. 初始化我们的全栈真实指标验证器
    print("⚙️ 初始化三项评估器 (MatterSim / DimeNet++ / CSLLM)...")
    evaluator = MetricsEvaluator(
        dimenet_weights_path=r'D:\Programming Software\github_project\MachineLearning_MG\models\weights\dimenet_best_ocp.pth'
    )

    # 4. 执行核心跑分逻辑
    results_df = evaluator.run_full_evaluation(atoms_list)

    # 将原本的序号 Material_ID 替换为你真实的 CIF 文件名，方便你对应！
    results_df['Material_ID'] = valid_cif_names

    # 保存 CSV 报告
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'external_metrics_report.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"💾 评估跑分数据已保存至: {csv_path}")

    # 5. 生成精美的可视化图表
    vis = Visualizer(output_dir=output_dir)
    vis.generate_all_plots(results_df, atoms_list)

    print("\n🎉 外部材料评估全部完成！请在 `results_external/` 目录下查看以下交付结果：")
    print(" ├─ external_metrics_report.csv (包含所有文件的详细三项跑分)")
    print(" ├─ her_performance.png (这 52 个材料的 HER 催化活性分布)")
    print(" ├─ stability_curve.png (稳定性与合成性的分布曲线)")
    print(" └─ generated_structures.png (在这 52 个中挑出的前 4 名极品材料渲染图)")


if __name__ == "__main__":
    main()