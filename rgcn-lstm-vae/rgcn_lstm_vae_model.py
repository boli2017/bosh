import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 定义 RGCN 层
class RGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGCNLayer, self).__init__()
        # 假设我们只有一种类型的边（出和入）
        self.rgcn = RGCNConv(in_channels, out_channels, num_relations=2)

    def forward(self, x, edge_index, edge_type):
        # x: [num_nodes, in_channels], edge_index: [2, num_edges]
        return self.rgcn(x, edge_index, edge_type)

# 定义 LSTM 层
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x: [batch, sequence_length, input_size]
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

# 定义 VAE
class VAE(nn.Module):
    def __init__(self, input_size, latent_size, num_nodes, num_features_per_node):
        super(VAE, self).__init__()
        self.num_nodes = num_nodes
        self.num_features_per_node = num_features_per_node
        # 编码器
        self.fc1 = nn.Linear(input_size, latent_size * 2)  # 输出均值和对数方差
        # 解码器
        self.fc2 = nn.Linear(latent_size, num_nodes * num_features_per_node)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码器
        h = F.relu(self.fc1(x))
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        # 解码器
        reconstructed_x = torch.sigmoid(self.fc2(z))
        # 重塑输出以匹配原始数据形状
        reconstructed_x = reconstructed_x.view(-1, self.num_nodes, self.num_features_per_node)
        return reconstructed_x, mu, log_var

# 将 RGCN, LSTM 和 VAE 组合
class RGCN_LSTM_VAE(nn.Module):
    def __init__(self, rgcn_in_channels, rgcn_out_channels, lstm_hidden_size, vae_latent_size, num_nodes, num_features_per_node):
        super(RGCN_LSTM_VAE, self).__init__()
        self.rgcn = RGCNLayer(rgcn_in_channels, rgcn_out_channels)
        self.lstm = LSTMEncoder(rgcn_out_channels * num_nodes, lstm_hidden_size)
        self.vae = VAE(lstm_hidden_size, vae_latent_size, num_nodes, num_features_per_node)

    def forward(self, x, edge_index, edge_type):
        sequence_length, num_nodes, num_features = x.shape

        # 处理每个时间步
        rgcn_outputs = []
        for t in range(sequence_length):
            rgcn_out = self.rgcn(x[t, :, :], edge_index, edge_type)
            if torch.isnan(rgcn_out).any():
                print(f"NaN detected in RGCN output at time step {t}")
            rgcn_outputs.append(rgcn_out.view(1, -1))

        # 将列表转换为张量
        rgcn_outputs = torch.cat(rgcn_outputs, dim=0)
        if torch.isnan(rgcn_outputs).any():
            print("NaN detected in RGCN outputs after concatenation")

        # LSTM 层
        lstm_in = rgcn_outputs.unsqueeze(0)
        lstm_out, _, _ = self.lstm(lstm_in)

        # VAE 层
        vae_input = lstm_out.squeeze(0)
        vae_out, mu, log_var = self.vae(vae_input)
        vae_out = vae_out.view(sequence_length, num_nodes, num_features)

        return vae_out, mu, log_var

# 从CSV文件加载数据来训练
csv_file_path = 'micServiceDemo1201.csv'
# 前两行是节点和节点特征
df = pd.read_csv(csv_file_path, skiprows=[0, 1], header=None)

# 平滑插值
df = df.interpolate()

# 节点和节点特征信息
nodes = ['adservice', 'cartservice', 'checkoutservice', 'currencyservice', 'emailservice', 'frontend', 'paymentservice', 'productcatalogservice', 'recommendationservice', 'shippingservice']
node_features = ['NetworkReceiveBytes', 'PodWorkload(Ops)', 'MemoryUsage(Mi)', 'PodLatency(s)', 'PodSuccessRate(%)', 'CpuUsage(m)', 'NetworkTransmitBytes']

# 节点关系信息
graph_relations = [('frontend', 'adservice'), ('frontend', 'recommendationservice'), ('frontend', 'productcatalogservice'), ('frontend', 'cartservice'), ('frontend', 'shippingservice'), ('frontend', 'currencyservice'), ('frontend', 'checkoutservice'), ('recommendationservice', 'productcatalogservice'), ('checkoutservice', 'productcatalogservice'), ('checkoutservice', 'cartservice'), ('checkoutservice', 'shippingservice'), ('checkoutservice', 'currencyservice'), ('checkoutservice', 'paymentservice'), ('checkoutservice', 'emailservice')]

# 一些参数
rgcn_output_dim = 32
lstm_units = 64
latent_dim = 16
batch_size = 32
# 根据节点数调整
num_nodes = len(nodes)
# 根据关系数调整
num_relations = 1

# 提取时间戳和节点特征数据
#timestamps = df.iloc[0].values
timestamps = df.iloc[2:, 0].values

# 每个节点的特征数量
num_features_per_node = len(node_features)

# 为每个时间点创建特征矩阵
input_data = np.zeros((len(timestamps), len(nodes), num_features_per_node))

# 遍历每个时间点，提取特征数据
for t in range(len(timestamps)):
    for i, node in enumerate(nodes):
        start_col = i * num_features_per_node
        end_col = start_col + num_features_per_node
        node_data = df.iloc[t, start_col+1:end_col+1]
        input_data[t, i, :] = node_data.values

# 构造邻接矩阵
adjacency_matrix = np.zeros((len(nodes), len(nodes)))

for relation in graph_relations:
    src_index = nodes.index(relation[0])
    dest_index = nodes.index(relation[1])
    adjacency_matrix[src_index, dest_index] = 1

print(input_data.shape)
# 输出input_data.shape: (4368, 10, 7)
print(adjacency_matrix.shape)
# 输出adjacency_matrix.shape: (10, 10)

# 将数据转换为torch.Tensor
feature_matrix = torch.tensor(input_data, dtype=torch.float)
adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float)

# 将邻接矩阵转换为边索引
edge_index, _ = dense_to_sparse(adjacency_matrix)

# 先简单地创建一个全零的边类型向量
edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

# 遍历每条边，确定其类型
for i in range(edge_index.size(1)):
    src, dst = edge_index[:, i]
    if adjacency_matrix[src][dst] == 1 and adjacency_matrix[dst][src] == 0:
        # 出边
        edge_type[i] = 0
    elif adjacency_matrix[src][dst] == 0 and adjacency_matrix[dst][src] == 1:
        # 入边
        edge_type[i] = 1
    # 如果两个方向都有边，可以添加额外的逻辑来处理这种情况
# 开始训练
# 数据清洗，缩放一些数值到0～1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(feature_matrix.numpy().reshape(-1, num_features_per_node))
scaled_data = scaled_data.reshape(-1, num_nodes, num_features_per_node)
feature_matrix = torch.tensor(scaled_data, dtype=torch.float)
# 数据划分 前2/3为输入，后2/3而输出
half_index = feature_matrix.shape[0] // 2
train_data = feature_matrix[:half_index]
test_data = feature_matrix[half_index:]

# 将数据输入到模型中
model = RGCN_LSTM_VAE(7, 16, 64, 10, num_nodes, num_features_per_node)

output = model(feature_matrix, edge_index, edge_type)
# 输出output.shape: (10, 10, 7)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)  # 学习率

# 训练参数
epochs = 100
batch_size = 32
train_total_losses = []
train_kl_losses = []
train_reconstruction_losses = []

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    kl_loss = 0
    reconstruction_loss = 0
    for i in range(0, train_data.shape[0], batch_size):
        # 获取批次数据
        batch_train = train_data[i:i+batch_size]
        batch_test = test_data[i:i+batch_size]

        # 前向传播
        outputs, mu, log_var = model(batch_train, edge_index, edge_type)

        # 重构损失
        recon_loss = criterion(outputs, batch_test)
        # KL散度损失
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # 总损失
        loss = recon_loss + kl_div
        total_loss += loss.item()
        kl_loss += kl_div.item()
        reconstruction_loss += recon_loss.item()

        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # 记录平均损失
    # avg_total_loss = total_loss / (train_data.shape[0] / batch_size)
    # avg_kl_loss = kl_loss / (train_data.shape[0] / batch_size)
    # avg_reconstruction_loss = reconstruction_loss / (train_data.shape[0] / batch_size)

    train_total_losses.append(total_loss)
    train_kl_losses.append(kl_loss)
    train_reconstruction_losses.append(reconstruction_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss:.4f}, KL Loss: {kl_loss:.4f}, Reconstruction Loss: {reconstruction_loss:.4f}')

# 绘制多张图
plt.figure(figsize=(12, 8))

# 总损失
plt.subplot(2, 2, 1)
plt.plot(train_total_losses, label='Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Total Loss')
plt.legend()

# 重构损失
plt.subplot(2, 2, 2)
plt.plot(train_reconstruction_losses, label='Reconstruction Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Reconstruction Loss')
plt.legend()

# KL损失
plt.subplot(2, 2, 3)
plt.plot(train_kl_losses, label='KL Divergence Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('KL Divergence Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 保存模型
torch.save(model.state_dict(), 'rgcn_lstm_vae_model.pth')
