import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import metrics
from student_kd_1_1_210_virtualnode_bicross import StudentModel, ProDataset  # 确保你保存的模型文件为 model_student.py
import pickle
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy import stats  # 需要新增这个 import
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# 参数设置
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
NUMBER_EPOCHS = 2500
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
RNA_TEACHER_PATH = "./Model/model256_All_sf2_1_2.pth"
PROTEIN_TEACHER_PATH = "/ifs/home/huangzhijian/shensiyuan/1/AGAT-PPIS-master/Model/Full_model_45.pkl"
MODEL_SAVE_PATH = "./Model_210/"
SEED = 42

# 设定随机种子
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def flush(self):
        pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


# 模型训练一个 epoch
def train_one_epoch(model, data_loader, optimizer, criterion, alpha=0.5, label_noise_std=0.01):
    model.train()
    total_loss, pred_loss, kd_loss = 0.0, 0.0, 0.0

    for rna_data, protein_data, labels in data_loader:
        rna_data = rna_data.to(DEVICE)
        protein_data = protein_data.to(DEVICE)
        labels = labels.to(DEVICE).float()

        # 标签扰动（平滑）
        labels = labels + torch.empty_like(labels).normal_(mean=0.0, std=label_noise_std)

        optimizer.zero_grad()

        # 模型前向传播，包含两个模态的教师和学生特征
        preds, s_rna, t_rna, s_pro, t_pro, attn_weights = model(rna_data, protein_data, return_attention=True)


        # 监督损失
        loss_pred = criterion(preds.squeeze(), labels.squeeze())

        # 知识蒸馏损失
        loss_kd = model.kd_loss(s_rna, t_rna, s_pro, t_pro, alpha=alpha)

        # 总损失
        loss = loss_pred + alpha * loss_kd

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累积 loss
        total_loss += loss.item()
        pred_loss += loss_pred.item()
        kd_loss += loss_kd.item()

    # 返回 epoch 平均损失
    return total_loss / len(data_loader), pred_loss / len(data_loader), kd_loss / len(data_loader)


# 验证模型
def evaluate(model, data_loader, criterion, fold=0, epoch=0):
    model.eval()
    epoch_loss = 0.0
    y_true, y_pred = [], []
    attn_weights = None
    protein_data = None

    with torch.no_grad():
        for batch_idx, (rna_data, protein_data, labels) in enumerate(data_loader):
            rna_data = rna_data.to(DEVICE)
            protein_data = protein_data.to(DEVICE)
            labels = labels.to(DEVICE).float()

            preds, _, _, _, _, attn_weights = model(rna_data, protein_data, return_attention=True)
            loss = criterion(preds.squeeze(), labels.squeeze())

            epoch_loss += loss.item()
            y_true.extend(labels.cpu().numpy().flatten().tolist())
            y_pred.extend(preds.cpu().numpy().flatten().tolist())

            break  # 只取一个batch用于可视化

    return epoch_loss / len(data_loader), y_true, y_pred






# 分析评估结果
def analysis(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    
    # 皮尔逊相关和 p-value
    pcc, pcc_p_value = stats.pearsonr(y_true, y_pred)
    
    # 斯皮尔曼相关和 p-value
    scc, scc_p_value = stats.spearmanr(y_true, y_pred)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "pcc": pcc,
        "pcc_p_value": pcc_p_value,
        "scc": scc,
        "scc_p_value": scc_p_value,
    }
def safe_to_int(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    while isinstance(x, (list, np.ndarray)) and len(x) > 0:
        x = x[0]
    return int(x)

def train(train_df, valid_df, fold=0):
    train_loader = DataLoader(ProDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(ProDataset(valid_df), batch_size=BATCH_SIZE, shuffle=False)

    model = StudentModel(
        rna_teacher_path=RNA_TEACHER_PATH,
        protein_teacher_path=PROTEIN_TEACHER_PATH
    ).to(DEVICE)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=0.5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUMBER_EPOCHS, eta_min=1e-6)

    patience = 150
    counter = 0
    early_stop = False

    label_noise_std = 0.01
    max_alpha = 0.5
    warmup_epochs = 50

    best_pcc = -1
    best_epoch = -1
    best_score = None
    history = {
        "train_total": [], "train_pred": [], "train_kd": [],
        "val_total": [], "rmse": [], "mae": [], "pcc": [], "scc": [], "alpha": []
    }

    for epoch in range(NUMBER_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUMBER_EPOCHS}")
        alpha = max_alpha * ((epoch + 1) / warmup_epochs) ** 2 if epoch < warmup_epochs else max_alpha
        print(f"Alpha = {alpha:.4f}")

        train_total_loss, train_pred_loss, train_kd_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, alpha=alpha, label_noise_std=label_noise_std
        )

        val_loss, val_true, val_pred = evaluate(
            model, valid_loader, criterion, fold=fold, epoch=epoch
        )

        val_metrics = analysis(val_true, val_pred)

        print(f"Train Loss: {train_total_loss:.4f} | Supervised: {train_pred_loss:.4f} | KD: {train_kd_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f}")
        print(f"PCC: {val_metrics['pcc']:.4f} | SCC: {val_metrics['scc']:.4f}")

        history["train_total"].append(train_total_loss)
        history["train_pred"].append(train_pred_loss)
        history["train_kd"].append(train_kd_loss)
        history["val_total"].append(val_loss)
        history["rmse"].append(val_metrics['rmse'])
        history["mae"].append(val_metrics['mae'])
        history["pcc"].append(val_metrics['pcc'])
        history["scc"].append(val_metrics['scc'])
        history["alpha"].append(alpha)

        score = composite_score(val_metrics)

        # ✅ 保存综合得分最高的模型
        if best_epoch == -1 or score > best_score:
            best_score = score
            best_pcc = val_metrics["pcc"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(Model_Path, f"Fold{fold}_best_model.pth"))
            print(f"New best model saved at epoch {best_epoch} with Composite Score: {best_score:.4f}")
            counter = 0
        else:
            counter += 1
            print(f"[EarlyStopping] No improvement for {counter} epochs.")
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}")
                early_stop = True
                break  # 添加 break 加速退出

    # === 最终返回训练结果 ===
    if best_epoch != -1:
        # 在训练结束后再次评估 best 模型
        best_model_path = os.path.join(Model_Path, f"Fold{fold}_best_model.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            final_val_loss, final_true, final_pred = evaluate(model, valid_loader, criterion, fold=fold, epoch=best_epoch)
            final_metrics = analysis(final_true, final_pred)
            return best_pcc, best_epoch, final_metrics
        else:
            print(f"[Error] Best model not found at: {best_model_path}")
            return None
    else:
        print("[Error] No best epoch found during training.")
        return None






def plot_loss_curves(history, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_total"], label="Train Total Loss")
    plt.plot(history["train_pred"], label="Supervised Loss")
    plt.plot(history["train_kd"], label="KD Loss")
    plt.plot(history["val_total"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves (Fold {fold})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, f"loss_curve_fold{fold}.png"))
    plt.close()

def composite_score(metrics):
    # 目标：越高越好，因此对 RMSE 和 MAE 取负号
    pcc = metrics["pcc"]
    scc = metrics["scc"]
    rmse = -metrics["rmse"]
    mae = -metrics["mae"]
    
    # 可调权重：建议先简单平均，如需微调可改成 w1*pcc + w2*scc + w3*(-rmse) + w4*(-mae)
    return pcc + scc + rmse + mae

# K折交叉验证
def cross_validation(all_df, n_splits=5):
    set_seed()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    best_info_per_fold = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(all_df)):
        print(f"\n==== Fold {fold + 1} ====")
        train_df = all_df.iloc[train_idx]
        valid_df = all_df.iloc[valid_idx]

        best_pcc, best_epoch, final_metrics = train(train_df, valid_df, fold=fold+1)
        best_info_per_fold.append({
            "Fold": fold + 1,
            "Best_Epoch": best_epoch,
            "PCC": best_pcc,
            "RMSE": final_metrics["rmse"],
            "MAE": final_metrics["mae"],
            "SCC": final_metrics["scc"],
        })

    print("\n==== Summary of All Folds ====")
    for info in best_info_per_fold:
        print(
            f"Fold {info['Fold']}: Best Epoch {info['Best_Epoch']} | "
            f"PCC: {info['PCC']:.4f} | RMSE: {info['RMSE']:.4f} | "
            f"MAE: {info['MAE']:.4f} | SCC: {info['SCC']:.4f}"
        )

    avg_pcc = np.mean([info["PCC"] for info in best_info_per_fold])
    print(f"\nAverage PCC over {n_splits} folds: {avg_pcc:.4f}")

    # 保存 summary.csv
    summary_df = pd.DataFrame(best_info_per_fold)
    summary_csv_path = os.path.join(checkpoint_path, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")

    return avg_pcc

    
# 数据加载
def load_data(pkl_path):
    with open(pkl_path, "rb") as f:
        data_dict = pickle.load(f)

    # 移除不需要的样本
    for remove_id in []:
        data_dict.pop(remove_id, None)

    records = []
    for k, v in data_dict.items():
        records.append({
            "ID": k,
            "protein_sequence": v["protein_sequence"],
            "rna_sequence": v["rna_sequence"],
            "label": v["label"]
        })

    return pd.DataFrame(records)

# 主程序
if __name__ == "__main__":
    # 创建新的时间戳文件夹
    import time
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    checkpoint_path = os.path.normpath("./Log210/" + localtime)  # 新增 Log 目录
    os.makedirs(checkpoint_path, exist_ok=True)

    Model_Path = os.path.normpath(checkpoint_path + '/model')  # 更新模型保存路径
    os.makedirs(Model_Path, exist_ok=True)

    # 重定向输出到新的日志文件
    sys.stdout = Logger(os.path.join(checkpoint_path, 'training.log'))

    df = load_data("./Dataset_210/data_dict.pkl")
    cross_validation(df, n_splits=5)

    sys.stdout.log.close()
