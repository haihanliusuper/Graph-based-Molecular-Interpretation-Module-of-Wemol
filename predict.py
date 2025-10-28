import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import GetAdjacencyMatrix
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torch.nn import Sequential as Seq, Linear, ReLU
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_scatter import scatter
import os
import copy
from torch_geometric.explain import GNNExplainer
from torch_geometric.data import Batch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from rdkit import Chem  # ✅ Chem 是 rdkit 下的模块，不要从 rdkit.Chem 导入它

from rdkit.Chem.Draw import rdMolDraw2D  # 导入绘图模块

def one_hot_encoding(value, choices):
    return [1 if value == choice else 0 for choice in choices]

def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Unknown']
    if not hydrogens_implicit:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    degree_enc = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridization_enc = one_hot_encoding(str(atom.GetHybridization()),
                                         ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = atom_type_enc + degree_enc + formal_charge_enc + hybridization_enc + \
                          is_in_ring_enc + is_aromatic_enc + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector, dtype=np.float32)

def get_bond_features(bond, use_stereochemistry=True):
    bond_type_enc = one_hot_encoding(bond.GetBondType(),
                                     [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])
    is_conj_enc = [int(bond.GetIsConjugated())]
    is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + is_conj_enc + is_in_ring_enc

    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()),
                                           ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector, dtype=np.float32)


from rdkit.Chem import GetAdjacencyMatrix

def create_graph_data(smiles_list):
    data_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n_nodes = mol.GetNumAtoms()
        X = np.array([get_atom_features(atom) for atom in mol.GetAtoms()])
        X = torch.tensor(X, dtype=torch.float)

        rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        E = np.array([get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j))) for i, j in zip(rows, cols)])
        edge_attr = torch.tensor(E, dtype=torch.float)

        data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([0.0]))  # dummy label
        data_list.append(data)
    return data_list



# ----------------------- 注意力池化 -----------------------

# ----------------------- 模型定义 -----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, MessagePassing
from torch.nn import Linear, Dropout, ReLU, BatchNorm1d, Sequential as Seq


class AttentionPooling(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.attention_mechanism = nn.Linear(node_dim, 1)
        self.mask_layer = nn.Linear(node_dim, 1)  # 让掩码与节点特征相关（代替 num_nodes）

    def forward(self, node_feats, batch_idx):
        attn_scores = self.attention_mechanism(node_feats)         # [N, 1]
        mask_logits = self.mask_layer(node_feats)                  # [N, 1]
        node_mask = torch.sigmoid(mask_logits)                     # [N, 1]

        final_scores = attn_scores * node_mask                     # [N, 1]
        pooled = scatter(node_feats * final_scores, batch_idx,
                         dim=0, reduce="sum")

        return pooled, final_scores.squeeze(-1)

        # scores→[N]

# ------------------------------------------------------------------
# 1.  GCN
# ------------------------------------------------------------------
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2,
                 out_dim, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim,  hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.dropout = Dropout(dropout_rate)
        self.attention_pooling = AttentionPooling(hidden_dim2)    # ← 旧名字
        self.out = Linear(hidden_dim2, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x, attn = self.attention_pooling(x, batch)
        return self.out(x), attn

# ------------------------------------------------------------------
# 2.  GAT
# ------------------------------------------------------------------
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2,
                 out_dim, dropout_rate=0.5, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim1, heads=heads)
        self.conv2 = GATConv(hidden_dim1 * heads, hidden_dim2, heads=1)
        self.dropout = Dropout(dropout_rate)
        self.attention_pooling = AttentionPooling(hidden_dim2)
        self.out = Linear(hidden_dim2, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x, attn = self.attention_pooling(x, batch)
        return self.out(x), attn

# ------------------------------------------------------------------
# 3.  GIN
# ------------------------------------------------------------------
class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2,
                 out_dim, dropout_rate=0.5):
        super().__init__()
        self.mlp1 = Seq(Linear(in_dim, hidden_dim1),
                        ReLU(),
                        Linear(hidden_dim1, hidden_dim1))
        self.conv1 = GINConv(self.mlp1)

        self.mlp2 = Seq(Linear(hidden_dim1, hidden_dim2),
                        ReLU(),
                        Linear(hidden_dim2, hidden_dim2))
        self.conv2 = GINConv(self.mlp2)

        self.dropout = Dropout(dropout_rate)
        self.attention_pooling = AttentionPooling(hidden_dim2)
        self.out = Linear(hidden_dim2, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x, attn = self.attention_pooling(x, batch)
        return self.out(x), attn

# ------------------------------------------------------------------
# 4.  MPNN (基于 GCNConv 的简易实现)
# ------------------------------------------------------------------
class MPNN(MessagePassing):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2,
                 out_dim, dropout_rate=0.5):
        super().__init__(aggr="mean")
        self.lin = Linear(in_dim, hidden_dim1)
        self.conv1 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv2 = GCNConv(hidden_dim2, hidden_dim2)
        self.dropout = Dropout(dropout_rate)
        self.attention_pooling = AttentionPooling(hidden_dim2)
        self.out = Linear(hidden_dim2, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.lin(x))
        x = self.dropout(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x, attn = self.attention_pooling(x, batch)
        return self.out(x), attn

# ---------------------------------------------------------------------------
# 加载权重 / 预测
# ---------------------------------------------------------------------------
def load_model(Model, path, in_dim, h1, h2, out_dim, drop):
    m = Model(in_dim,h1,h2,out_dim,drop)
    m.load_state_dict(torch.load(path,map_location=device))
    return m.to(device).eval()

def predict(model, data_list, task):
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list(data_list).to(device)
        out = model(batch)           # tuple (pred, attn) or pred
        preds = out[0] if isinstance(out,tuple) else out
        if task=="classification":
            prob = F.softmax(preds,dim=1)[:,1]
            return prob.cpu().numpy().tolist(), (prob>=0.5).int().cpu().numpy().tolist()
        else:
            return preds.view(-1).cpu().numpy().tolist(), None

# ---------------------------------------------------------------------------
# Attention 可视化
# ---------------------------------------------------------------------------

import os
import copy
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Batch
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def save_attention_images(
        smiles_list,
        models: dict,          # {"GCN": model, ...}
        data_list,             # 与 smiles_list 对齐的 Data 列表
        base_dir: str = "images_attention"):
    """
    为每个模型 & 每个分子绘制原子级 attention 热图（使用红蓝色谱表示正负）。
    图像保存路径:  images_attention/<模型名>/mol_<idx>.png
    """
    for model_name, model in models.items():
        model.eval()
        device = next(model.parameters()).device
        out_dir = os.path.join(base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        for idx, (smi, data) in enumerate(zip(smiles_list, data_list)):
            d_copy = copy.deepcopy(data)
            batch = Batch.from_data_list([d_copy]).to(device)

            with torch.no_grad():
                out = model(batch)
                if not (isinstance(out, (tuple, list)) and len(out) == 2):
                    print(f"{model_name} 未返回 attn，跳过 idx={idx}")
                    continue
                _, attn = out

            attn = attn.cpu().numpy().flatten()

            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumAtoms() != len(attn):
                print(f"SMILES 无效或原子数不匹配，跳过 idx={idx}")
                continue

            # ---- 使用 red-blue 双向色图（负值蓝，正值红）----
            vmin, vmax = attn.min(), attn.max()
            vcenter = 0.0
            ε = 1e-6  # 极小偏移量

            if vmin >= vcenter:
                vmin = vcenter - ε
            if vmax <= vcenter:
                vmax = vcenter + ε

            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            cmap = cm.get_cmap('coolwarm')  # 蓝-白-红色图

            highlight_colors = {
                i: tuple(cmap(norm(w))[:3])  # RGB颜色
                for i, w in enumerate(attn)
            }
            highlight_radii = {i: 0.5 for i in range(len(attn))}

            # ---- RDKit 绘图 ----
            drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(highlight_colors.keys()),
                highlightAtomColors=highlight_colors,
                highlightAtomRadii=highlight_radii
            )
            drawer.FinishDrawing()
            img_path = os.path.join(out_dir, f"mol_{idx}.png")
            with open(img_path, "wb") as f:
                f.write(drawer.GetDrawingText())

    print(f"🧠 Attention 双向热图已保存至 {base_dir}/<模型名>/mol_X.png")
from torch_geometric.explain import Explainer, GNNExplainer
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, edge_index, batch):
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, batch=batch)
        out = self.model(data)
        return out[0] if isinstance(out, (tuple, list)) else out


from rdkit.Chem.Draw import rdMolDraw2D  # 别忘了放在 import 部分
import os, copy
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Batch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from torch_geometric.explain import Explainer, GNNExplainer

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch):
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, batch=batch)
        out = self.model(data)
        return out[0] if isinstance(out, (tuple, list)) else out


def save_gnnexplainer_images(
        smiles_list,
        models: dict,
        data_list,
        task_type: str = "classification",
        base_dir: str = "images_explainer",
        expl_epochs: int = 200):
    """
    使用 Explainer + GNNExplainer 为每个模型和分子生成因果热图。
    保存路径：images_explainer/<模型名>/mol_<idx>.png
    """

    for model_name, model in models.items():
        dev = next(model.parameters()).device
        out_dir = os.path.join(base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        model_cfg = dict(
            mode='multiclass_classification' if task_type == 'classification' else 'regression',
            task_level='graph',
            return_type='log_probs' if task_type == 'classification' else 'raw'
        )

        explainer = Explainer(
            model=WrappedModel(model),  # 使用代理模型
            algorithm=GNNExplainer(epochs=expl_epochs),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=model_cfg
        )

        for idx, (smi, data) in enumerate(zip(smiles_list, data_list)):
            d = copy.deepcopy(data)
            d.batch = torch.zeros(d.x.size(0), dtype=torch.long)  # 添加 batch 信息
            d = d.to(dev)

            try:
                explanation = explainer(x=d.x, edge_index=d.edge_index, batch=d.batch)
            except Exception as e:
                print(f"[{model_name}] idx={idx} GNNExplainer 失败：{e}")
                continue

            node_mask = explanation.get('node_mask')
            if node_mask is None:
                print(f"[{model_name}] idx={idx} 未生成 node_mask，跳过")
                continue

            score = node_mask.cpu().numpy().flatten()

            # 使用 coolwarm 色图 + 双向归一化
            vmin, vmax = score.min(), score.max()
            vcenter = 0.0
            ε = 1e-6
            if vmin >= vcenter:
                vmin = vcenter - ε
            if vmax <= vcenter:
                vmax = vcenter + ε

            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            cmap = cm.get_cmap("coolwarm")

            highlight_colors = {i: cmap(norm(s))[:3] for i, s in enumerate(score)}
            highlight_radii = {i: 0.5 for i in range(len(score))}

            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumAtoms() != len(score):
                print(f"[{model_name}] idx={idx} 原子数不匹配，跳过")
                continue

            drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(highlight_colors.keys()),
                highlightAtomColors=highlight_colors,
                highlightAtomRadii=highlight_radii
            )
            drawer.FinishDrawing()
            with open(os.path.join(out_dir, f"mol_{idx}.png"), "wb") as f:
                f.write(drawer.GetDrawingText())

    print(f"🎯 GNNExplainer 热图已保存至 {base_dir}/<模型名>/mol_X.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict_input", required=True, help="SMILES csv")
    ap.add_argument("--predict_col", default="Smiles")
    ap.add_argument("--output", default="predict_result.csv")
    ap.add_argument("--task_type", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--hidden_dim1", type=int, default=128)
    ap.add_argument("--hidden_dim2", type=int, default=256)
    ap.add_argument("--dropout_rate", type=float, default=0.5)
    ap.add_argument("--expl_epochs", type=int, default=200,
                    help="GNNExplainer 训练轮数")
    args = ap.parse_args()

    # 1. 读取 SMILES → 图
    df = pd.read_csv(args.predict_input)
    smiles = df[args.predict_col].tolist()
    data_list = create_graph_data(smiles)
    in_dim  = data_list[0].num_node_features
    out_dim = 2 if args.task_type == "classification" else 1

    # 2. 加载四个模型
    models = {
        "GCN":  load_model(GCN,  "./best_GCN_model.pth",
                           in_dim, args.hidden_dim1, args.hidden_dim2,
                           out_dim, args.dropout_rate),
        "GAT":  load_model(GAT,  "./best_GAT_model.pth",
                           in_dim, args.hidden_dim1, args.hidden_dim2,
                           out_dim, args.dropout_rate),
        "GIN":  load_model(GIN,  "./best_GIN_model.pth",
                           in_dim, args.hidden_dim1, args.hidden_dim2,
                           out_dim, args.dropout_rate),
        "MPNN": load_model(MPNN, "./best_MPNN_model.pth",
                           in_dim, args.hidden_dim1, args.hidden_dim2,
                           out_dim, args.dropout_rate)
    }

    # 3. 批量预测
    results = {"Smiles": smiles}
    for name, model in models.items():
        sc, lb = predict(model, data_list, args.task_type)
        results[f"{name}_Score"] = sc
        if lb is not None:
            results[f"{name}_Label"] = lb
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"✅ 预测完成：结果写入 {args.output}")

    # 4. Attention 热图
    save_attention_images(smiles, models, data_list,
                          base_dir="images_attention")

    # 5. GNNExplainer 因果热图（新增调用）
    save_gnnexplainer_images(smiles, models, data_list,
                             task_type=args.task_type,
                             base_dir="images_explainer",
                             expl_epochs=args.expl_epochs)