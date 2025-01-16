import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head

        # (nums_head * head_dim = hidden_dim)
        assert hidden_dim % nums_head == 0
        self.head_dim = hidden_dim // nums_head

        self.dropout = nn.Dropout(dropout_rate)

        # Define linear projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        # x has shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.size()

        # Q, K, V each has shape: (batch_size, seq_len, hidden_dim)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshaping from (batch_size, seq_len, hidden_dim) to (batch_size, seq_len, nums_head, head_dim)
        # Then transpose to (batch_size, nums_head, seq_len, head_dim)
        # q_state = Q.view(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)  # [Another approach to do it]
        q = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        k = K.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        v = V.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)

        # Matrix multiplication: (batch_size, nums_head, seq_len, head_dim) * (batch_size, nums_head, head_dim, seq_len)
        # Resulting shape: (batch_size, nums_head, seq_len, seq_len)
        # Note that the scaling factor uses head_dim, not hidden_dim.
        attention_val = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        print(f"attention_val shape is {attention_val.size()}")
        print(f"attention_mask shape is {attention_mask.size()}")
        if attention_mask is not None:
            # If attention_mask is provided, it should have shape (batch_size, nums_head, seq_len, seq_len).
            assert attention_val.size() == attention_mask.size()
            attention_val = torch.masked_fill(attention_val, attention_mask == 0, float("-inf"))

        # Apply softmax along the last dimension to get attention weights.
        attention_weight = torch.softmax(attention_val, dim=-1)
        
        # Dropout on attention weights
        attention_weight = self.dropout(attention_weight)
        
        # Multiply attention weights with V:
        # (batch_size, nums_head, seq_len, seq_len) * (batch_size, nums_head, seq_len, head_dim)
        # -> (batch_size, nums_head, seq_len, head_dim)
        output_tmp = attention_weight @ v

        # Transpose back: (batch_size, nums_head, seq_len, head_dim)
        # -> (batch_size, seq_len, nums_head, head_dim)
        # -> (batch_size, seq_len, hidden_dim)
        #
        # Note: The transpose operation changes the dimension ordering but does not change the memory layout,
        # resulting in a non-contiguous tensor. The contiguous() method makes the tensor contiguous in memory,
        # allowing subsequent view or reshape operations without error.
        output_tmp = output_tmp.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        # output = output_mid.permute(0, 2, 1, 3).reshpae(batch_size, seq_len, self.hidden_dim)  # # [Another approach to do it]

        output = self.output_proj(output_tmp)
        return output


if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    batch_size, seq_len, hidden_dim = x.size()
    nums_head = 2

    # attention_mask has shape: (batch_size, nums_head, seq_len, seq_len).
    # Here we use a lower-triangular mask to simulate causal masking.
    attention_mask = torch.tril(torch.ones(batch_size, nums_head, seq_len, seq_len))
    print(attention_mask)

    multi_head_attention = MultiHeadAttention(hidden_dim=hidden_dim, nums_head=nums_head)
    
    x_forward = multi_head_attention.forward(x, attention_mask=attention_mask)
    print(x_forward)
    print(x_forward.size())
