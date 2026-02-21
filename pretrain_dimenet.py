import os
import torch
import lmdb
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

# 引入我们在模型层定义的 DimeNet++ =

from models.dimnet_model import build_dimenet_plus_plus


# ==========================================
# 1. 定义 OCP LMDB 数据集读取器 (适配 PyG)
# ==========================================
class OCPLmdbDataset(Dataset):
    def __init__(self, lmdb_path):
        super().__init__()
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
            self.keys = [key for key, _ in txn.cursor()]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data_bytes = txn.get(self.keys[idx])
            data_obj = pickle.loads(data_bytes)

        # 1. 绕过 PyG 版本限制
        data_dict = data_obj.__dict__
        if '_store' in data_dict:
            source_dict = data_dict['_store']
        else:
            source_dict = data_dict

        # 2. 提取原子序数和坐标 (适配不同版本的 OCP)
        raw_z = source_dict.get('atomic_numbers', source_dict.get('z'))
        raw_pos = source_dict['pos']

        # 3. 智能寻找目标能量标签 (适配 IS2RE 任务的 y_relaxed)
        if 'y_relaxed' in source_dict:
            raw_y = source_dict['y_relaxed']
        elif 'y' in source_dict:
            raw_y = source_dict['y']
        else:
            # 如果还是找不到，打印出所有可用的键，方便我们排查
            raise KeyError(f"数据集中找不到目标能量标签。当前可用的键有: {list(source_dict.keys())}")

        # 4. 安全地转换为 Tensor
        def safe_tensor(val, dtype):
            if isinstance(val, torch.Tensor):
                return val.clone().detach().to(dtype)
            return torch.tensor(val, dtype=dtype)

        z = safe_tensor(raw_z, torch.long)
        pos = safe_tensor(raw_pos, torch.float)
        y = safe_tensor(raw_y, torch.float)

        # 确保吸附能 y 是一个 1D Tensor
        if y.dim() == 0:
            y = y.unsqueeze(0)
        elif y.numel() > 1:
            y = y.view(-1)[0].unsqueeze(0)

        return Data(z=z, pos=pos, y=y)


# ==========================================
# 2. 核心训练与验证循环
# ==========================================
def train_dimenet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数设置
    batch_size = 32
    epochs = 10
    learning_rate = 1e-4
    # 使用 r'' 防止 Windows 路径反斜杠转义
    lmdb_train_path = r'D:\Programming Software\github_project\MachineLearning_MG\dataset\is2re\10k\train\data.lmdb'
    save_dir = 'models/weights/'
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    print("Loading OCP Dataset...")
    train_dataset = OCPLmdbDataset(lmdb_train_path)
    # 实际应用中需要划分 train 和 val，这里为演示简化
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型与优化器
    model = build_dimenet_plus_plus(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 能量预测通常使用 L1 Loss (MAE) 作为标准评估指标
    criterion = torch.nn.L1Loss()

    best_loss = float('inf')

    print("Starting Pre-training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # DimeNet++ 前向传播 (传入原子序数, 坐标, 和 batch 索引)
            preds = model(batch.z, batch.pos, batch.batch)

            # 计算损失 (预测吸附能 vs 真实吸附能)
            loss = criterion(preds.squeeze(), batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            pbar.set_postfix({'MAE Loss (eV)': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch + 1} Completed | Average Train MAE: {avg_loss:.4f} eV")

        # ==========================================
        # 3. 保存最优模型权重
        # ==========================================
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, 'dimenet_best_ocp.pth')

            # 推荐保存 state_dict 而不是整个模型，这样更灵活且跨版本兼容
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"🌟 新的最优权重已保存至: {save_path}")


if __name__ == "__main__":
    train_dimenet()