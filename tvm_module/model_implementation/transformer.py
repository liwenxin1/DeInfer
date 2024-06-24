import torch
import torch.nn as nn
import tvm
from tvm import relay
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x):
        return self.attention(x, x, x)[0]

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# # Example usage
# d_model = 512
# nhead = 8
# num_layers = 6
# d_ff = 2048

# encoder = TransformerEncoder(num_layers, d_model, nhead, d_ff)

# # Input shape (batch_size, sequence_length, d_model)
# batch_size = 32
# sequence_length = 50
# input_data = torch.rand((batch_size, sequence_length, d_model))

# target = tvm.target.Target("cuda")
# scripted_model = torch.jit.trace(encoder, input_data)
# input_name = "data"
# shape_list = [(input_name, input_data.shape)]
# relay_module, relay_params = relay.frontend.from_pytorch(scripted_model, shape_list)
# with relay.build_config(opt_level=3):
#     lib= relay.build(relay_module, target=target, params=relay_params)
    
# lib.export_library("tvm_module/model_implementation/transformer.so")
    
