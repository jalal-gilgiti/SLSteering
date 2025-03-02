# Model 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # x: [batch_size, hidden_size] or [batch_size, seq_len, hidden_size]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        return x + self.pe[:, :x.size(1)].to(x.device)

class AdvancedQueryOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_hints=20, num_outputs=1, num_heads=8):
        super(AdvancedQueryOptimizer, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.hint_output = nn.Linear(hidden_size, num_hints)
        self.cost_output = nn.Linear(hidden_size, num_outputs)

    def forward(self, x, attn_bias=None):
        # x: [batch_size, input_size]
        x = self.input_proj(x)  # [batch_size, hidden_size]
        x = self.positional_encoding(x)  # [batch_size, 1, hidden_size]
        x = self.transformer_encoder(x)  # [batch_size, 1, hidden_size]
        x = x.squeeze(1)  # [batch_size, hidden_size]
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        hint_pred = self.hint_output(x)  # [batch_size, 20], logits
        cost_pred = self.cost_output(x)  # [batch_size, 1]
        return hint_pred, cost_pred
    


# Model 2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LSTM

class QueryOptimizerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_hints, num_outputs=1, num_heads=8):
        super(QueryOptimizerModel, self).__init__()

        # Transformer Encoder for High-Level Feature Extraction
        self.encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=4)
        
        # Multihead Attention to Learn Query Relations
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_size)  # New normalization layer for attention residual

        # BiLSTM for Temporal & Sequential Learning
        self.bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_linear = nn.Linear(hidden_size * 2, hidden_size)  # Compress BiLSTM output
        self.norm_lstm = nn.LayerNorm(hidden_size)  # New normalization after LSTM

        # Fully Connected Layers with Layer Normalization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.3)

        # Output Layers
        self.hint_output = nn.Linear(hidden_size, num_hints)  # Hint classification
        self.cost_output = nn.Linear(hidden_size, num_outputs)  # Cost regression

    def forward(self, queryformer_embeddings):
        # Initial Feature Transformation
        x = self.fc1(queryformer_embeddings)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Multihead Attention for Query Structure
        x = x.unsqueeze(1)  # Convert to (batch_size, 1, hidden_size)
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.norm_attn(x + attn_output)  # Residual connection with normalization
        x = x.squeeze(1)  # Back to (batch_size, hidden_size)

        # Transformer Encoder for Query Refinement (with residual connection and activation)
        x = x.unsqueeze(1)
        encoded_x = self.transformer_encoder(x)
        x = self.relu(x + encoded_x)  # Residual connection with ReLU activation
        x = x.squeeze(1)

        # BiLSTM for Sequential Learning (with Residual Connection)
        x = x.unsqueeze(1)
        lstm_out, _ = self.bilstm(x)
        x = self.norm_lstm(self.lstm_linear(lstm_out.squeeze(1)) + x.squeeze(1))  # Residual connection with normalization

        # Fully Connected Layers
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)

        # Output Predictions
        hint_pred = self.hint_output(x)  # Hint classification
        cost_pred = self.cost_output(x)  # Cost regression

        return hint_pred, cost_pred