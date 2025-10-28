import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import argparse
import torch

import torch
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch.nn import Linear, Dropout, ReLU

from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
from torch.utils.data import random_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
import torch.nn as nn
from torch_scatter import scatter
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import PeriodicTable
from torch.utils.data import DataLoader, random_split

from rdkit.Chem import GetAdjacencyMatrix
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
def one_hot_encoding(value, choices):
    """ One-hot encodes the given value based on a list of choices. """
    encoding = [1 if choice == value else 0 for choice in choices]
    return encoding

def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br',  'Unknown']
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                               "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry=True):
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    data_list = []
    for (smiles, y_val) in zip(x_smiles, y):
        mol = Chem.MolFromSmiles(smiles)
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
        X = torch.tensor(X, dtype=torch.float)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))#####链接矩阵定义
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)
        EF = np.zeros((n_edges, n_edge_features))
        for (k, (i, j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
        EF = torch.tensor(EF, dtype=torch.float)
        y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)
        data_list.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor))
    return data_list



# 解析命令行参数
parser = argparse.ArgumentParser(description="Train and evaluate GNN models")
parser.add_argument("--hidden_dim1", type=int, default=128, help="Size of the first hidden layer")
parser.add_argument("--hidden_dim2", type=int, default=256, help="Size of the second hidden layer")
parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
parser.add_argument("--output_pdf", type=str, default="model_comparison_report.pdf", help="Output PDF file name")
#parser.add_argument("--input", type=str, default="esol.csv", help="Path to the input CSV file")
parser.add_argument("--ptinput", type=str, default="esol_data.pt", help="Path to the input .pt file")

args = parser.parse_args()

import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib

matplotlib.use('Agg')  # 设置后端为 'Agg'
import matplotlib.pyplot as plt
# 加载数据集
ptinput = args.ptinput
loaded_data = torch.load(ptinput)  # 假设保存了 data_list 和 x_smiles
data_list = loaded_data['data_list']
x_smiles = loaded_data['x_smiles']

# 检查数据集的节点特征维度
num_features = data_list[0].num_node_features
print(f"Number of node features: {num_features}")

# 划分训练集和测试集
total_data = len(data_list)
test_size = int(0.2 * total_data)
train_size = total_data - test_size
train_data, test_data = random_split(data_list, [train_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        return pooled




# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.out = torch.nn.Linear(hidden_dim2, output_dim)

        # 实例化 AttentionPooling 模块
        self.attention_pooling = AttentionPooling(hidden_dim2)  # 使用 hidden_dim2 作为 node_dim

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # 调用 AttentionPooling 的 forward 方法
        x = self.attention_pooling(x, batch)  # 注意这里传递的是 x 和 batch
        return self.out(x)


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim1, heads=heads)
        self.conv2 = GATConv(hidden_dim1 * heads, hidden_dim2, heads=1)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.out = torch.nn.Linear(hidden_dim2, output_dim)

        # 实例化 AttentionPooling 模块
        self.attention_pooling = AttentionPooling(hidden_dim2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # 使用 AttentionPooling 替代 global_mean_pool
        x = self.attention_pooling(x, batch)
        return self.out(x)

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5):
        super(GIN, self).__init__()
        self.mlp1 = Seq(Linear(input_dim, hidden_dim1), ReLU(), Linear(hidden_dim1, hidden_dim1))
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = Seq(Linear(hidden_dim1, hidden_dim2), ReLU(), Linear(hidden_dim2, hidden_dim2))
        self.conv2 = GINConv(self.mlp2)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.out = torch.nn.Linear(hidden_dim2, output_dim)

        # 实例化 AttentionPooling 模块
        self.attention_pooling = AttentionPooling(hidden_dim2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # 使用 AttentionPooling 替代 global_mean_pool
        x = self.attention_pooling(x, batch)
        return self.out(x)


class MPNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5):
        super(MPNN, self).__init__(aggr='mean')
        self.lin = Linear(input_dim, hidden_dim1)
        self.conv1 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv2 = GCNConv(hidden_dim2, hidden_dim2)
        self.dropout = Dropout(dropout_rate)
        self.out = Linear(hidden_dim2, output_dim)
        self.relu = ReLU()

        # 实例化 AttentionPooling 模块
        self.attention_pooling = AttentionPooling(hidden_dim2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.lin(x))
        x = self.dropout(x)
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))

        # 使用 AttentionPooling 替代 global_mean_pool
        x = self.attention_pooling(x, batch)
        return self.out(x)
# 加载数据集

# 检查数据集的节点特征维度
num_features = data_list[0].num_node_features
print(f"Number of node features: {num_features}")

# 划分训练集和测试集
total_data = len(data_list)
test_size = int(0.2 * total_data)
train_size = total_data - test_size
train_data, test_data = random_split(data_list, [train_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# 训练和测试函数
def train(model, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze(1)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, criterion):
    model.eval()
    total_loss = 0
    predictions = []
    true_values = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data).squeeze(1)
            loss = criterion(output, data.y)
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            true_values.extend(data.y.cpu().numpy())
    r2 = r2_score(true_values, predictions)
    return total_loss / len(test_loader), r2


# 模型训练和评估
def train_and_evaluate(ModelClass, model_name):
    model = ModelClass(
        input_dim=num_features,
        hidden_dim1=args.hidden_dim1,  # 使用命令行参数
        hidden_dim2=args.hidden_dim2,  # 使用命令行参数
        output_dim=1,
        dropout_rate=args.dropout_rate  # 使用命令行参数
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    train_losses = []
    test_losses = []
    r2_scores = []
    best_r2 = -float('inf')

    start_time = time.time()
    for epoch in range(100):
        train_loss = train(model, optimizer, criterion)
        test_loss, r2 = test(model, criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        r2_scores.append(r2)

        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), f'best_{model_name}_model.pth')

        print(
            f'{model_name} - Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, R2: {r2:.4f}')

    training_time = (time.time() - start_time) / 60  # 转换为分钟
    print(f'{model_name} training completed in {training_time:.2f} minutes.')

    return {
        'model_name': model_name,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'r2_scores': r2_scores,
        'best_r2': best_r2,
        'training_time': training_time
    }


# 运行所有模型
results = {}
models = {'GCN': GCN, 'GAT': GAT, 'GIN': GIN, 'MPNN': MPNN}
for name, model_class in models.items():
    results[name] = train_and_evaluate(model_class, name)


# 生成报告
def generate_report(results):
    with PdfPages(args.output_pdf) as pdf:
        # 对比表格
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        table_data = []
        for model_name, result in results.items():
            table_data.append([
                model_name,
                f"{result['best_r2']:.4f}",
                f"{result['training_time']:.2f} min"
            ])
        table = ax.table(
            cellText=table_data,
            colLabels=['Model', 'Best R²', 'Training Time'],
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title('Model Comparison Summary')
        pdf.savefig()
        plt.close()

        # 训练和测试损失曲线
        for model_name, result in results.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(result['train_losses'], label='Train Loss')
            ax.plot(result['test_losses'], label='Test Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{model_name} - Training and Testing Loss')
            ax.legend()
            pdf.savefig()
            plt.close()

        # R²分数曲线
        fig, ax = plt.subplots(figsize=(10, 5))
        for model_name, result in results.items():
            ax.plot(result['r2_scores'], label=model_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison')
        ax.legend()
        pdf.savefig()
        plt.close()

        # 最优时的散点图
        for model_name, result in results.items():
            # 加载最佳模型
            model = models[model_name](
                input_dim=num_features,
                hidden_dim1=args.hidden_dim1,
                hidden_dim2=args.hidden_dim2,
                output_dim=1,
                dropout_rate=args.dropout_rate
            ).to(device)
            model.load_state_dict(torch.load(f'best_{model_name}_model.pth'))
            model.eval()

            # 获取预测值和真实值
            predictions = []
            true_values = []
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    output = model(data).squeeze(1)
                    predictions.extend(output.cpu().numpy())
                    true_values.extend(data.y.cpu().numpy())

            # 绘制散点图
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(true_values, predictions, alpha=0.5)
            ax.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title(f'{model_name} - Best R²: {result["best_r2"]:.4f}')
            pdf.savefig()
            plt.close()

# 生成报告
generate_report(results)
print("Report saved as 'model_comparison_report.pdf'.")