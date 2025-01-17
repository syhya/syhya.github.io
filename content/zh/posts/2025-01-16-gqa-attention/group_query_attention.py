import math
import torch
import torch.nn as nn


class GQABroadcast(nn.Module):
    """
    Group Query Attention (GQA) implementation:
    By configuring `nums_kv_head` (G, the number of groups), this module supports:
      - When nums_kv_head == nums_head: Multi-Head Attention (MHA)
      - When nums_kv_head == 1: Multi-Query Attention (MQA)
      - When 1 < nums_kv_head < nums_head: Generic Grouped Query Attention (GQA)
    """
    def __init__(self, hidden_dim, nums_head, nums_kv_head, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head  # Total number of Q heads (H)
        self.nums_kv_head = nums_kv_head # Number of K, V heads (G, groups)
        assert hidden_dim % nums_head == 0
        assert nums_head % nums_kv_head == 0

        self.head_dim = hidden_dim // nums_head
        # Number of Q heads per group
        self.q_heads_per_group = nums_head // nums_kv_head
        self.dropout = nn.Dropout(dropout_rate)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # Projection output dimensions for K, V = nums_kv_head * head_dim
        self.k_proj = nn.Linear(hidden_dim, nums_kv_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_kv_head * self.head_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask= None):
        batch_size, seq_len, _ = x.size()
        Q = self.q_proj(x)  # (batch_size, seq_len, hidden_dim)
        K = self.k_proj(x)  # (batch_size, seq_len, nums_kv_head * head_dim)
        V = self.v_proj(x)  # (batch_size, seq_len, nums_kv_head * head_dim)

        # Q: (batch_size, seq_len, hidden_dim)
        # -> (batch_size, seq_len, nums_head, head_dim)
        # -> (batch_size, nums_head, seq_len, head_dim)
        # -> (batch_size, nums_kv_head, q_heads_per_group, seq_len, head_dim)
        q = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2).contiguous()
        q = q.view(batch_size, self.nums_kv_head, self.q_heads_per_group, seq_len, self.head_dim)

        # K, V: (batch_size, seq_len, nums_kv_head * head_dim)
        #  -> (batch_size, seq_len, nums_kv_head, head_dim)
        # -> (batch_size, nums_kv_head, seq_len, head_dim
        # -> (batch_size, nums_kv_head, 1, seq_len, head_dim)
        k = K.view(batch_size, seq_len, self.nums_kv_head, self.head_dim).transpose(1, 2).unsqueeze(2)
        v = V.view(batch_size, seq_len, self.nums_kv_head, self.head_dim).transpose(1, 2).unsqueeze(2)

        # q: (batch_size, nums_kv_head, q_heads_per_group, seq_len, head_dim) * (batch_size, nums_kv_head, 1, head_dim, seq_len)
        # -> (batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len)
        attention_val = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        # mask
        if attention_mask is not None:
            attention_val = torch.masked_fill(attention_val, attention_mask == 0, float("-inf"))

        # softmax
        attention_weight = torch.softmax(attention_val, dim=-1)

        # dropout
        attention_weight = self.dropout(attention_weight)

        # (batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len) * (batch_size, nums_kv_head, 1, seq_len, head_dim)
        # -> (batch_size, nums_kv_head, q_heads_per_group, seq_len, head_dim)
        output_tmp = attention_weight @ v

        # (batch_size, nums_kv_head, q_heads_per_group, seq_len, head_dim)
        # -> (batch_size, nums_head, seq_len, head_dim)
        output_tmp = output_tmp.view(batch_size, self.nums_head, seq_len, self.head_dim)

        # (batch_size, nums_head, seq_len, head_dim)
        # -> (batch_size, seq_len, nums_head, head_dim) -> (batch_size, seq_len, hidden_dim)
        output_concat = output_tmp.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_proj(output_concat)
        return output


class GQARepeat(nn.Module):
    """
    Group Query Attention (GQA) implementation:
    By configuring `nums_kv_head` (G, the number of groups), this module supports:
      - When nums_kv_head == nums_head: Multi-Head Attention (MHA)
      - When nums_kv_head == 1: Multi-Query Attention (MQA)
      - When 1 < nums_kv_head < nums_head: Generic Grouped Query Attention (GQA)
    """
    def __init__(self, hidden_dim, nums_head, nums_kv_head, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_kv_head = nums_kv_head
        assert hidden_dim % nums_head == 0
        assert nums_head % nums_kv_head == 0
        self.head_dim = hidden_dim // nums_head
        self.q_head_per_group = nums_head // nums_kv_head

        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_kv_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_kv_head * self.head_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        # (batch_size, seq_len, hidden_dim)
        Q = self.q_proj(x)
        # (batch_size, seq_len, nums_kv_head * self.head_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # -> (batch_size, seq_len, nums_head, head_dim)
        # -> (batch_size, nums_head, seq_len, head_dim)
        q = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)

        # -> (batch_size, seq_len, nums_kv_head, head_dim)
        # -> (batch_size, nums_kv_head, seq_len, head_dim)
        k = K.view(batch_size, seq_len, self.nums_kv_head, self.head_dim).transpose(1, 2)
        v = V.view(batch_size, seq_len, self.nums_kv_head, self.head_dim).transpose(1, 2)

        # (batch_size, nums_head, seq_len, head_dim)
        k_repeat = k.repeat_interleave(self.q_head_per_group, dim=1)
        v_repeat = v.repeat_interleave(self.q_head_per_group, dim=1)

        # (batch_size, nums_head, seq_len, seq_len)
        attention_val = q @ k_repeat.transpose(-1, -2) / math.sqrt(self.head_dim)

        # mask
        if attention_mask is not None:
            attention_val = torch.masked_fill(attention_val, attention_mask == 0, float('-inf'))
        
        # softmax
        attention_weight = torch.softmax(attention_val, dim=-1)

        # dropout
        attention_weight = self.dropout(attention_weight)

        # (batch_size, nums_head, seq_len, head_dim)
        output_tmp = attention_weight @ v_repeat

        # (batch_size, seq_len, hidden_dim)
        output_concat = output_tmp.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        output = self.output_proj(output_concat)
        return output


if __name__ == "__main__":
    x = torch.randn(2, 3, 16)
    batch_size, seq_len, hidden_dim = x.size()
    nums_head = 8
    head_dim = hidden_dim // nums_head
    nums_kv_head = 4
    q_heads_per_group = nums_head // nums_kv_head
    
    # v1: Boardcast
    # attention_mask_v1 has shape: (batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len)
    attention_mask_v1 = torch.tril(torch.ones(batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len))
    gqa_boradcast = GQABroadcast(hidden_dim=hidden_dim, nums_head=nums_head,
                                                nums_kv_head=nums_kv_head, dropout_rate=0.1)
    x_forward_v1 = gqa_boradcast.forward(x, attention_mask=attention_mask_v1)

    # print(x_forward_v1)
    print(x_forward_v1.size())

    # v2: Repeat
    # attention_mask_v2 has shape: (batch_size, nums_head, seq_len, seq_len)
    attention_mask_v2 = torch.tril(torch.ones(batch_size, nums_head, seq_len, seq_len))
    gqa_repeat = GQARepeat(hidden_dim=hidden_dim, nums_head=nums_head,
                                                nums_kv_head=nums_kv_head, dropout_rate=0.1)
    x_forward_v2 = gqa_repeat.forward(x, attention_mask=attention_mask_v2)

    # print(x_forward_v2)
    print(x_forward_v2.size())



