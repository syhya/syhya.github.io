import torch
import torch.nn as nn
import math


class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        assert hidden_dim % nums_head == 0
        self.head_dim = hidden_dim // nums_head

        self.dropout = nn.Dropout(p=dropout)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)        
        # For kv, project: hidden_dim -> head_dim
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * 1)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * 1)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()

        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)

        # Broadcast k and v to match q's dimensions for attention computation
        # k -> (batch_size, 1, seq_len, head_dim)
        # v -> (batch_size, 1, seq_len, head_dim)
        k = K.unsqueeze(1)
        v = V.unsqueeze(1)

        # (batch_size, head_num, seq_len, head_dim) * (batch_size, 1, head_dim, seq_len)
        # -> (batch_size, head_num, seq_len, seq_len)
        attention_val = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        print(f"attention_val  shape is {attention_val.size()}")

        if attention_mask is not None:
            attention_val = torch.masked_fill(attention_val, attention_mask == 0, float("-inf"))
          
        attention_weight = torch.softmax(attention_val, dim=-1)
        print(f"attention_weight is {attention_weight}")
        attention_weight = self.dropout(attention_weight)

        # (batch_size, head_num, seq_len, seq_len) * (batch_size, 1, seq_len, head_dim)
        # -> (batch_size, head_num, seq_len, head_dim)
        output_tmp = attention_weight @ v

        # -> (batch_size, seq_len, head_num, head_dim)
        # -> (batch_size, seq_len, hidden_dim)
        output_tmp = output_tmp.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_proj(output_tmp)
        return output


if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    batch_size, seq_len, hidden_dim = x.size()
    nums_head = 2
    attention_mask = torch.tril(torch.ones(batch_size, nums_head, seq_len, seq_len))
    print(attention_mask)

    multi_query_attention = MultiQueryAttention(hidden_dim=hidden_dim, nums_head=nums_head, dropout=0.2)
    
    x_forward = multi_query_attention.forward(x, attention_mask=attention_mask)
    print(x_forward)
    print(x_forward.size())

        








