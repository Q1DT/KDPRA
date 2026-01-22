import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATConv
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn.utils.rnn as rnn_utils

MAP_CUTOFF = 10
Feature_Path = "./Feature_210/"
# 定义设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义特征函数
def rna_embedding(sequence_name):
    esm_feature = np.load("./Feature_210/RNA_embedding/representations/" + sequence_name + '.npy')
    return esm_feature.astype(np.float32)

def protein_embedding(sequence_name):
    dssp_feature = np.load("./Feature_210/DSSP_210/" + sequence_name + '_dssp.npy')   # [N, 14]
    esm_feature = np.load("./Feature_210/protein_esm3B/" + sequence_name + '.npy')    # [N, 2560]
    virtual_node_feat = np.load("./Feature_210/virtual_node_feats/" + sequence_name + '.npy')  # [1, 2574]
    return esm_feature.astype(np.float32), dssp_feature.astype(np.float32), virtual_node_feat.astype(np.float32)


def cal_edges(sequence_name, sequence_type='rna', radius=MAP_CUTOFF):
    if sequence_type == 'protein':
        dist_matrix_path = "./Dataset_210/蛋白质_distance_matrices/" + sequence_name + '_distance_matrix.npy'
    elif sequence_type == 'rna':
        dist_matrix_path = "./Dataset_210/RNA_distance_matrices/" + sequence_name + '_distance_matrix.npy'
    else:
        raise ValueError("sequence_type must be 'protein' or 'rna'.")

    dist_matrix = np.load(dist_matrix_path)
    mask = ((dist_matrix >= 0) & (dist_matrix <= radius))
    adjacency_matrix = mask.astype(int)
    radius_index_list = np.where(adjacency_matrix == 1)

    return [list(nodes) for nodes in radius_index_list]


# 修改后的 ProDataset
class ProDataset(Dataset):
    def __init__(self, dataframe,
                 protein_psepos_path='./Dataset_210/protein_dict.pkl',
                 rna_psepos_path='./Dataset_210/rna_dict.pkl'):
        self.rna_names = dataframe['ID'].values
        self.rna_sequences = dataframe['rna_sequence'].values

        self.protein_names = dataframe['ID'].values  # 默认蛋白质ID与RNA一致，如不同需改为 dataframe['protein_id']
        self.protein_sequences = dataframe['protein_sequence'].values

        self.labels = dataframe['label'].values

        self.rna_residue_psepos = pickle.load(open(rna_psepos_path, 'rb'))
        self.protein_residue_psepos = pickle.load(open(protein_psepos_path, 'rb'))


    def __getitem__(self, index):
        # 提取 RNA 信息
        rna_name = self.rna_names[index]
        rna_sequence = self.rna_sequences[index]
        rna_embed = rna_embedding(rna_name)

        # RNA 节点类型特征 (这里假设为全0，或可改为实际类型编码)
        rna_x = torch.zeros(len(rna_sequence), 1, dtype=torch.long)
        rna_emb = torch.from_numpy(rna_embed).float()

        # RNA 边
        rna_radius_index_list = cal_edges(rna_name, sequence_type='rna')
        rna_edge_index = torch.tensor(rna_radius_index_list, dtype=torch.long)

        # 构建 RNA 图
        rna_data = Data(
            x=rna_x,
            emb=rna_emb,
            edge_index=rna_edge_index,
            rna_len=[len(rna_sequence)],  # 需要 list 类型支持 batch
            id=self.rna_names[index]
        )

        # 提取蛋白质信息（此处假设与 RNA 共用同一 ID）
        protein_name = rna_name  # 默认 ID 一致
        esm_feat, dssp_feat, virtual_feat = protein_embedding(protein_name) # 分开拿出 [N, 2560], [N, 14]

        protein_x = torch.zeros(len(esm_feat), 1, dtype=torch.long)
        protein_emb = {
            "esm": torch.from_numpy(esm_feat).float(),         # [N, 2560]
            "dssp": torch.from_numpy(dssp_feat).float(),       # [N, 14]
            "virtual": torch.from_numpy(virtual_feat).float()  # [1, 2574]
        }
        # 蛋白质边
        protein_radius_index_list = cal_edges(protein_name, sequence_type='protein')
        protein_edge_index = torch.tensor(protein_radius_index_list, dtype=torch.long)

        # 构建蛋白质图
        protein_data = Data(
            x=protein_x,
            emb=protein_emb,  # 传 dict，后续模型融合用
            edge_index=protein_edge_index,
            protein_len=[len(esm_feat)],
            id=self.protein_names[index]
        )

        # 标签
        label = torch.tensor([self.labels[index]], dtype=torch.float32)

        return rna_data, protein_data, label

    def __len__(self):
        return len(self.labels)
# RNA教师模型特征提取器 (定义补充)
class RNA_feature_extraction(nn.Module):
    def __init__(self, hidden_size, hidden_dim=256):
        super(RNA_feature_extraction, self).__init__()

        self.CNN = CNN(hidden_size)

        self.emb_to_gat = nn.Linear(640, hidden_size)
        self.conv1 = GATConv(hidden_size, hidden_dim, heads=4, dropout=0.1, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, dropout=0.1, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_size, dropout=0.1, concat=False)

        self.x_embedding = nn.Embedding(6, hidden_size)

        self.line1 = nn.Linear(128, hidden_size)
        self.hidden_size = hidden_size

        self.line_emb = nn.Linear(640, hidden_size)
        self.line_g = nn.Linear(128, 128)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        emb = data.emb

        x_r = self.x_embedding(x[:, 0].int())
        x_g = self.emb_to_gat(emb)
        x_g = self.relu(x_g)

        x = self.relu(self.conv1(x_g, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))

        emb_graph = global_mean_pool(x, data.batch)
        emb = F.relu(self.line_emb(emb))

        # === 修复后的 rna_len 解析 ===
        if isinstance(data.rna_len, torch.Tensor):
            node_lens = data.rna_len.view(-1).tolist()
        elif isinstance(data.rna_len, list):
            if isinstance(data.rna_len[0], (list, torch.Tensor)):  # 嵌套 list 或 Tensor
                node_lens = [int(x) for sub in data.rna_len for x in (sub.tolist() if isinstance(sub, torch.Tensor) else sub)]
            else:
                node_lens = [int(x) for x in data.rna_len]
        else:
            node_lens = [int(data.rna_len)]

        # === 分段构造序列特征和 mask ===
        flag = 0
        out_graph, out_seq, out_r, mask = [], [], [], []

        for count_i in node_lens:
            count_i = int(count_i)
            mask.append([1] * count_i + [0] * (512 - count_i))

            x1 = x[flag:flag + count_i]
            x1 = torch.cat((x1, torch.zeros((512 - count_i, self.hidden_size), device=x.device)), dim=0)
            out_graph.append(x1)

            emb1 = emb[flag:flag + count_i]
            emb1 = torch.cat((emb1, torch.zeros((512 - count_i, self.hidden_size), device=x.device)), dim=0)
            out_seq.append(emb1)

            x_r1 = x_r[flag:flag + count_i]
            x_r1 = torch.cat((x_r1, torch.zeros((512 - count_i, self.hidden_size), device=x.device)), dim=0)
            out_r.append(x_r1)

            flag += count_i

        out_graph = torch.stack(out_graph).to(x.device)
        out_seq = torch.stack(out_seq).to(x.device)
        out_r = torch.stack(out_r).to(x.device)
        mask_seq = torch.tensor(mask, dtype=torch.float, device=x.device)

        out_r = (out_r + out_seq) / 2
        out_seq_cnn = self.CNN(out_r)
        emb_seq = (out_seq_cnn * mask_seq.unsqueeze(2)).mean(dim=1)

        return None, None, None, None, None, emb_seq



# 学生RNA特征提取器
class StudentRNAFeatureExtractor(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size

        self.emb_to_gat = nn.Linear(640, hidden_size)

        self.gat1 = GATConv(hidden_size, hidden_size, heads=4, concat=False)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.gat2 = GATConv(hidden_size, hidden_size, heads=4, concat=False)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.gat3 = GATConv(hidden_size, hidden_size, concat=False)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.cnn = CNN(hidden_size)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, rna_data):
        x = self.emb_to_gat(rna_data.emb)
        x = self.relu(x)

        residual = x
        x = self.gat1(x, rna_data.edge_index)
        x = self.norm1(x + residual)
        x = self.relu(x)

        residual = x
        x = self.gat2(x, rna_data.edge_index)
        x = self.norm2(x + residual)
        x = self.relu(x)

        residual = x
        x = self.gat3(x, rna_data.edge_index)
        x = self.norm3(x + residual)
        x = self.relu(x)

        graph_feat = global_mean_pool(x, rna_data.batch)
        cnn_feat = self.cnn(x.unsqueeze(0)).mean(dim=1)

        fusion_feat = (graph_feat + cnn_feat) / 2
        enhanced_feat = self.fusion_mlp(fusion_feat)

        return enhanced_feat, x  # x 是残基级特征 [N, D]


# # 蛋白质特征提取器
# class ProteinFeatureExtractor(nn.Module):
#     def __init__(self, in_dim=2560, hidden_dim=256):
#         super().__init__()
#         self.gat1 = GATConv(in_dim, hidden_dim, heads=4)
#         self.gat2 = GATConv(hidden_dim*4, hidden_dim, heads=1)

#     def forward(self, protein_data):
#         x = F.relu(self.gat1(protein_data.x, protein_data.edge_index))
#         x = F.relu(self.gat2(x, protein_data.edge_index))
#         return global_mean_pool(x, protein_data.batch)

# CNN (教师模型一致)
class CNN(nn.Module):
    def __init__(self, hidden_size):
        super(CNN, self).__init__()
        kernel_size = [7, 11, 15]

        self.conv_xt_1 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size[0], padding=(kernel_size[0]-1)//2)
        self.conv_xt_2 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size[1], padding=(kernel_size[1]-1)//2)
        self.conv_xt_3 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size[2], padding=(kernel_size[2]-1)//2)

        self.fc1_xt = nn.Linear(128, 128)

        self.line1 = nn.Linear(hidden_size // 2, 512)
        self.line2 = nn.Linear(512, hidden_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x1 = self.conv_xt_1(x)
        x2 = self.conv_xt_2(x)
        x3 = self.conv_xt_3(x)

        x = (x1 + x2 + x3) / 3
        x = x.permute(0, 2, 1)

        x = self.line2(self.dropout(self.relu(self.line1(x))))
        return x


# 分类器
class Classifier(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.layers(x)

    
# 教师模型中用于提取蛋白质特征的编码器（结构必须与训练用时一致）
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, dropout):
        super(GraphEncoder, self).__init__()
        self.gat = GATConv(input_dim, hidden_dim, heads=n_heads, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim * n_heads,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.gat(x, edge_index)  # [num_nodes, hidden_dim * heads]
        x = self.dropout(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)  # [num_nodes, hidden_dim * heads]
        return x


class ProteinTeacherModel(nn.Module):
    def __init__(self, model_path=None, input_dim=2560, hidden_dim=32, heads=4, dropout=0.2, out_dim=256):
        super(ProteinTeacherModel, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, heads, dropout)
        self.projector = nn.Linear(hidden_dim * heads, out_dim)  # Align to student output dim (256)

        if model_path is not None:
            self._load_encoder_weights(model_path)

    def forward(self, protein_data):
        x = protein_data.emb["esm"]  # 只使用 ESM 特征（教师模型不需要 DSSP）
        edge_index = protein_data.edge_index
        batch = protein_data.batch

        x = self.encoder(x, edge_index, batch)
        x = global_mean_pool(x, batch)
        return self.projector(x)  # Now returns [batch_size, 256]

    def _load_encoder_weights(self, path):
        print(f"[Info] Loading encoder weights from {path}")
        full_state_dict = torch.load(path, map_location='cpu')

        encoder_state = {
            k.replace("protein_encoder.", ""): v
            for k, v in full_state_dict.items()
            if k.startswith("protein_encoder.")
        }

        missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state, strict=False)

        if missing_keys:
            print("[Warning] Missing keys in protein_teacher encoder:", missing_keys)
        if unexpected_keys:
            print("[Warning] Unexpected keys in protein_teacher encoder:", unexpected_keys)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class ProteinStudentModel(nn.Module):
    def __init__(self, input_dim=2560, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size

        self.emb_to_hidden = nn.Linear(input_dim, hidden_size)

        self.gat1 = GATConv(hidden_size, hidden_size, heads=4, concat=False)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.gat2 = GATConv(hidden_size, hidden_size, heads=4, concat=False)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.gat3 = GATConv(hidden_size, hidden_size, concat=False)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.dssp_proj = nn.Linear(14, 2560)          # 投影 DSSP 到 2560
        self.virtual_proj = nn.Linear(2574, 14)       # 将 virtual_feats 映射为 DSSP 维度

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, protein_data):
        esm = protein_data.emb["esm"]         # [N, 2560]
        dssp = protein_data.emb["dssp"]       # [N, 14]
        virtual = protein_data.emb["virtual"] # [1, 2574]

        # === Debug 检查维度 ===
        # print("[Debug] esm.shape:", esm.shape)
        # print("[Debug] dssp.shape:", dssp.shape)
        # print("[Debug] virtual.shape:", virtual.shape)

        # 确保 virtual 为 2D，形状为 [1, 2574]
        if virtual.dim() == 1:
            virtual = virtual.unsqueeze(0)
        elif virtual.size(0) != 1:
            virtual = virtual[:1]

        # 将 virtual 特征投影到 DSSP 同维度
        virtual_feat = self.virtual_proj(virtual)  # [1, 14]

        # 广播给每个残基：将虚拟节点加到每个位置
        virtual_feat_broadcasted = virtual_feat.expand(dssp.size(0), -1)  # [N, 14]

        # 融合 DSSP + virtual
        dssp_combined = dssp + virtual_feat_broadcasted  # [N, 14]
        dssp_proj = self.dssp_proj(dssp_combined)        # [N, 2560]

        # 融合 ESM + DSSP
        x = esm + dssp_proj  # [N, 2560]

        # === GNN 主干 ===
        edge_index = protein_data.edge_index
        batch = protein_data.batch

        x = self.emb_to_hidden(x)             # [N, hidden_size]
        x = self.relu(x)

        residual = x
        x = self.gat1(x, edge_index)
        x = self.norm1(x + residual)
        x = self.relu(x)

        residual = x
        x = self.gat2(x, edge_index)
        x = self.norm2(x + residual)
        x = self.relu(x)

        residual = x
        x = self.gat3(x, edge_index)
        x = self.norm3(x + residual)
        x = self.relu(x)

        graph_feat = global_mean_pool(x, batch)         # [B, hidden_size]
        enhanced_feat = self.mlp(graph_feat)            # [B, hidden_size]

        return enhanced_feat, x  # x: [N, hidden_size] 残基级特征



class ResidueCrossAttention(nn.Module):
    def __init__(self, embed_dim=256, heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, rna_res_feat, protein_res_feat, rna_mask=None, protein_mask=None, return_attn=False):
        attn_output, attn_weights = self.multihead_attn(
            query=rna_res_feat,
            key=protein_res_feat,
            value=protein_res_feat,
            key_padding_mask=protein_mask,
            need_weights=True,
            average_attn_weights=False  # 保留每个 head 的注意力
        )

        x = self.norm(rna_res_feat + self.dropout(attn_output))
        x = self.norm(x + self.ffn(x))

        if return_attn:
            return x, attn_weights  # shape: [B, H, L_q, L_k]
        return x


class StudentModel(nn.Module):
    def __init__(self, rna_teacher_path, protein_teacher_path):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256)
        )

        self.rna_teacher = RNA_feature_extraction(hidden_size=256)
        full_state_dict = torch.load(rna_teacher_path, map_location=DEVICE)
        rna_state_dict = {
            k.replace("rna_graph_model.", ""): v
            for k, v in full_state_dict.items()
            if k.startswith("rna_graph_model.")
        }
        self.rna_teacher.load_state_dict(rna_state_dict)
        self.rna_teacher.eval()
        for p in self.rna_teacher.parameters():
            p.requires_grad = False

        self.rna_student = StudentRNAFeatureExtractor()
        self.protein_teacher = ProteinTeacherModel(protein_teacher_path)
        self.protein_teacher.eval()
        for p in self.protein_teacher.parameters():
            p.requires_grad = False

        self.protein_student = ProteinStudentModel()
        self.cross_attention = ResidueCrossAttention(embed_dim=256)
        self.classifier = Classifier(input_dim=256)

    def _batch_to_padded_sequence(self, node_feats, batch):
        B = int(batch.max().item()) + 1
        sequences = []
        for i in range(B):
            mask = (batch == i)
            sequences.append(node_feats[mask])
        padded = rnn_utils.pad_sequence(sequences, batch_first=True)  # [B, L, D]
        return padded
        
    def forward(self, rna_data, protein_data=None, return_attention=False, fusion_strategy="mean"):
        with torch.no_grad():
            _, _, _, _, _, rna_teacher_feat = self.rna_teacher(rna_data)
            if protein_data is not None:
                protein_teacher_feat = self.protein_teacher(protein_data)
            else:
                protein_teacher_feat = torch.zeros_like(rna_teacher_feat)

        # 学生模型输出
        rna_student_feat, rna_node_feat = self.rna_student(rna_data)

        if protein_data is not None:
            protein_student_feat, pro_node_feat = self.protein_student(protein_data)

            # 转换成 padded 残基序列
            rna_seq = self._batch_to_padded_sequence(rna_node_feat, rna_data.batch)      # [B, L_rna, D]
            pro_seq = self._batch_to_padded_sequence(pro_node_feat, protein_data.batch)  # [B, L_pro, D]

            # 双向交叉注意力
            rna_to_pro = self.cross_attention(rna_seq, pro_seq)        # RNA ← Protein
            pro_to_rna = self.cross_attention(pro_seq, rna_seq)        # Protein ← RNA

            # 融合策略选择
            if fusion_strategy == "mean":
                fused_feat = (rna_to_pro.mean(dim=1) + pro_to_rna.mean(dim=1)) / 2       # [B, D]
            elif fusion_strategy == "concat":
                concat_feat = torch.cat([rna_to_pro.mean(dim=1), pro_to_rna.mean(dim=1)], dim=1)  # [B, 2D]
                fused_feat = self.fusion_mlp(concat_feat)  # MLP 降维到 D
            else:
                raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")
        else:
            # 没有蛋白质信息，直接使用 RNA 学生向量
            fused_feat = rna_student_feat
            protein_student_feat = None
            protein_teacher_feat = torch.zeros_like(rna_student_feat)

        affinity_pred = self.classifier(fused_feat)

        if return_attention:
            if protein_data is not None:
                _, attn_weights = self.cross_attention(rna_seq, pro_seq, return_attn=True)
            else:
                attn_weights = None
            return affinity_pred, rna_student_feat, rna_teacher_feat, protein_student_feat, protein_teacher_feat, attn_weights
        return affinity_pred, rna_student_feat, rna_teacher_feat, protein_student_feat, protein_teacher_feat

    def kd_loss(self, student_rna, teacher_rna, student_protein, teacher_protein, alpha=0.5):
        cos = nn.CosineSimilarity(dim=1)
        loss_rna = 1 - cos(student_rna, teacher_rna).mean()
        loss_pro = 1 - cos(student_protein, teacher_protein).mean()
        return (loss_rna + loss_pro) / 2
