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

# Load data and create graph data list
# 解析参数
import argparse
import pandas as pd
from rdkit import Chem
import torch

parser = argparse.ArgumentParser(description="构建图数据并保存为 .pt 文件")
parser.add_argument("--input_csv", type=str, required=True, help="输入 CSV 路径")
parser.add_argument("--output_pt", type=str, required=True, help="输出 .pt 文件路径")
parser.add_argument("--smiles_col", type=str, default="Smiles", help="SMILES 列名")
parser.add_argument("--label_col", type=str, default="Label", help="标签列名")
args = parser.parse_args()

# 读取 CSV 文件
dataframe = pd.read_csv(args.input_csv)

# 强制转换为字符串，并清除 NaN
dataframe[args.smiles_col] = dataframe[args.smiles_col].astype(str).fillna("").str.strip()

# 筛选合法 SMILES
def is_valid_smiles(s):
    try:
        return Chem.MolFromSmiles(s) is not None
    except:
        return False

valid_mask = dataframe[args.smiles_col].apply(is_valid_smiles)
invalid_rows = dataframe[~valid_mask]
if not invalid_rows.empty:
    print("⚠️ 检测到非法或无效 SMILES 行，将被忽略：")
    print(invalid_rows[[args.smiles_col, args.label_col]])

# 保留合法数据
dataframe = dataframe[valid_mask]

# 转为列表
x_smiles = dataframe[args.smiles_col].tolist()
y_labels = dataframe[args.label_col].tolist()

# 调用函数生成图数据
data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y_labels)

# 保存
torch.save({'data_list': data_list, 'x_smiles': x_smiles}, args.output_pt)
print(f"✅ 数据集和 SMILES 已保存到: {args.output_pt}")
