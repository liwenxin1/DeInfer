import torch
import torch.nn as nn

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM层
        out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        return out

# # 定义输入维度、隐藏层维度、LSTM层数和输出维度
# input_size = 100
# hidden_size = 200
# num_layers = 10
# output_size = 1

# # 创建LSTM模型
# model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# # 打印模型结构
# print(model)

# # 定义输入数据
# batch_size = 100
# seq_length = 100
# input_data = torch.randn(batch_size, seq_length, input_size)

# # 前向传播
# output = model(input_data)

# # 打印输出的形状
# print("Output shape:", output.shape)