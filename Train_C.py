import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import random
from torch.nn import Sequential as Seq, Linear, ReLU
matplotlib.use('Agg')  # 设置后端为 'Agg'
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch.nn import Linear, Dropout, ReLU

from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import argparse
# 注意力池化模块

###形参
parser = argparse.ArgumentParser(description="Model Training and Evaluation")
# 添加隐藏层维度的参数
parser = argparse.ArgumentParser(description="Model Training and Evaluation")
parser.add_argument("--hidden_dim1", type=int, default=128, help="Dimension of the first hidden layer")
parser.add_argument("--hidden_dim2", type=int, default=256, help="Dimension of the second hidden layer")
parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate") 
parser.add_argument("--ptinput", type=str, default="esol_data.pt", help="Path to the input .pt file")  # 💡 改名以对称
parser.add_argument("--output_pdf", type=str, default="model_performance_summary2.pdf", help="Path to the output PDF file")


args = parser.parse_args()

data_file_path = args.ptinput
output_pdf_path = args.output_pdf
hidden_dim1 = args.hidden_dim1
hidden_dim2 = args.hidden_dim2

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


# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.dropout = torch.nn.Dropout(dropout_rate)  # 添加Dropout层
        self.out = torch.nn.Linear(hidden_dim2, output_dim)
        self.attention_pooling = AttentionPooling(hidden_dim2)  # 添加注意力池化模块

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)  # 在第一个卷积层后添加Dropout
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)  # 在第二个卷积层后添加Dropout

        # 使用注意力池化代替全局均值池化
        x = self.attention_pooling(x, batch)

        x = self.out(x)
        return x

# 定义 GAT 模型
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim1, heads=heads)
        self.conv2 = GATConv(hidden_dim1 * heads, hidden_dim2, heads=1)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.out = torch.nn.Linear(hidden_dim2, output_dim)
        self.attention_pooling = AttentionPooling(hidden_dim2)  # 添加注意力池化模块

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.attention_pooling(x, batch)  # 使用注意力池化
        x = self.out(x)
        return x

# 定义 GIN 模型
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


# 定义 MPNN 模型
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
loaded_data = torch.load(data_file_path)
data_list = loaded_data['data_list']
x_smiles = loaded_data['x_smiles']

# 检查数据集的节点特征维度
num_features = data_list[0].num_node_features
print(f"Number of node features: {num_features}")

# 假设我们有2个类别
num_classes = 2

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

# 初始化模型、优化器和损失函数
models = {
    'GCN': GCN(input_dim=num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=num_classes, dropout_rate=0.7),
    'GAT': GAT(input_dim=num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=num_classes, dropout_rate=0.7),
    'GIN': GIN(input_dim=num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=num_classes, dropout_rate=0.7),
    'MPNN': MPNN(input_dim=num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=num_classes, dropout_rate=0.7)
}
all_labels = torch.cat([data.y for data in train_loader]).cpu().numpy()
class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# 更新损失函数，加入类别权重
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# 训练函数
def train(model, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 测试函数
def test(model, loader):
    model.eval()
    total_loss = 0
    predictions = []
    true_values = []
    losses = []
    y_scores = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.long())
            total_loss += loss.item()

            batch_loss = torch.nn.functional.cross_entropy(output, data.y.long(), reduction='none')
            losses.extend(batch_loss.cpu().numpy())

            preds = output.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_values.extend(data.y.cpu().numpy())

            if output.shape[1] == 2:
                y_score = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            else:
                y_score = torch.softmax(output, dim=1).cpu().numpy()
            y_scores.extend(y_score)

    accuracy = accuracy_score(true_values, predictions)
    cm = confusion_matrix(true_values, predictions)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    class_loss = {}
    unique_classes = np.unique(true_values)
    for c in unique_classes:
        mask = np.array(true_values) == c
        class_loss[c] = np.mean(np.array(losses)[mask]) if sum(mask) > 0 else 0.0

    roc_auc = None
    if len(unique_classes) == 2:
        fpr, tpr, _ = roc_curve(true_values, y_scores)
        roc_auc = auc(fpr, tpr)

    return {
        'total_loss': total_loss / len(loader),
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'class_losses': class_loss,
        'roc_auc': roc_auc,
        'y_scores': y_scores,
        'true_values': true_values,
        'predictions': predictions
    }

optimizers = {name: torch.optim.Adam(model.parameters(), lr=0.001) for name, model in models.items()}
# 训练和测试所有模型
all_models_results = {}
for model_name in models:
    model = models[model_name].to(device)
    optimizer = optimizers[model_name]

    train_losses = []
    test_losses = []
    best_accuracy = 0.0

    for epoch in range(10):
        train_loss = train(model, optimizer)
        test_results = test(model, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_results['total_loss'])

        if test_results['accuracy'] > best_accuracy:
            best_accuracy = test_results['accuracy']
            torch.save(model.state_dict(), f'best_{model_name}_model.pth')

        print(
            f'Model: {model_name}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_results["total_loss"]:.4f}, Accuracy: {test_results["accuracy"]:.4f}')

    model.load_state_dict(torch.load(f'best_{model_name}_model.pth'))
    final_results = test(model, test_loader)
    all_models_results[model_name] = final_results

# 创建 PDF 文件
with PdfPages(output_pdf_path) as pdf:
    for model_name, results in all_models_results.items():
        # 绘制训练和测试损失曲线
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Training and Testing Loss')
        plt.legend()
        pdf.savefig()
        plt.close()

        # 绘制 ROC 曲线（如果是二分类问题）
        if results['roc_auc'] is not None:
            fpr, tpr, _ = roc_curve(results['true_values'], results['y_scores'])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            pdf.savefig()
            plt.close()

        # 添加一句话总结准确度
        summary_text = f"{model_name} Best Model Accuracy: {results['accuracy']:.4f}\n"
        for i, acc in enumerate(results['class_accuracies']):
            summary_text += f"Class {i} Accuracy: {acc:.4f}\n"
        plt.figure(figsize=(8, 2))
        plt.text(0.1, 0.5, summary_text, fontsize=12, va='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()

print("Training complete. All plots saved to 'model_performance_summary.pdf'.")