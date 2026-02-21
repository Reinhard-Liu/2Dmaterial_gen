import os
import matplotlib.pyplot as plt
import seaborn as sns
from ase.visualize.plot import plot_atoms


class Visualizer:
    # 默认直接输出到 results 文件夹，满足你的要求
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # 设置全局绘图风格
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['Arial']

    def plot_delta_g_distribution(self, df):
        """绘制 ΔG_H 分布直方图 (her_performance.png)"""
        plt.figure(figsize=(8, 6))
        sns.histplot(df['Delta_G_H (eV)'], bins=20, kde=True, color='teal')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Ideal Activity (0 eV)')
        plt.title('HER Performance ($\Delta G_H$)', fontsize=14)
        plt.xlabel('$\Delta G_H$ (eV)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend()
        save_path = os.path.join(self.output_dir, 'her_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_stability_curve(self, df):
        """绘制 Formation Energy 与 Stability Score 的散点/曲线关系 (stability_curve.png)"""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='Formation_Energy (eV/atom)', y='Stability_Score',
                        hue='Is_Synthesizable', palette='Set2', s=100, alpha=0.8)
        plt.title('Stability vs Synthesizability Score', fontsize=14)
        plt.xlabel('Formation Energy ($E_{form}$)', fontsize=12)
        plt.ylabel('Stability Score', fontsize=12)
        save_path = os.path.join(self.output_dir, 'stability_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def render_3d_structures(self, atoms_list, top_k=4):
        """渲染排名前列的 3D 原子结构球棍模型 (generated_structures.png)"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < len(atoms_list):
                # 调用 ASE 的绘图引擎，展示俯视图 (XY平面)
                plot_atoms(atoms_list[i], ax, radii=0.8, rotation=('0x,0y,0z'))
                ax.set_title(f'Generated 2D Material #{i + 1}', fontsize=12)
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'generated_structures.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_all_plots(self, df, atoms_list):
        print(f"🎨 正在生成并保存可视化图表至 {self.output_dir} ...")
        self.plot_delta_g_distribution(df)
        self.plot_stability_curve(df)

        # 挑选最接近 0 eV 的顶级材料进行 3D 渲染
        top_indices = df['Abs_Delta_G_H (eV)'].nsmallest(4).index
        top_atoms = [atoms_list[idx] for idx in top_indices]
        self.render_3d_structures(top_atoms)
        print("✅ 效果图保存完毕！")