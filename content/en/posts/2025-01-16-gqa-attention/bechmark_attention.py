import math
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------
# Group Query Attention (GQA) Implementation
# -------------------
class GroupQueryAttention(nn.Module):
    """
    Group Query Attention (GQA) implementation:
    By configuring `nums_kv_head` (G, the number of groups), this module supports:
      - When nums_kv_head == nums_head: Multi-Head Attention (MHA)
      - When nums_kv_head == 1: Multi-Query Attention (MQA)
      - When 1 < nums_kv_head < nums_head: Generic Grouped Query Attention (GQA)
    """
    def __init__(self, hidden_dim, nums_head, nums_kv_head, dropout=0.1):
        """
        Args:
            hidden_dim (int): Dimensionality of the input embeddings.
            nums_head (int): Total number of heads for Q (H).
            nums_kv_head (int): Number of heads/groups for K and V (G).
            dropout (float): Dropout probability for attention weights.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head  # total Q heads
        self.nums_kv_head = nums_kv_head  # total K, V heads (grouped)
        assert hidden_dim % nums_head == 0, "hidden_dim must be divisible by nums_head."
        assert nums_head % nums_kv_head == 0, "nums_head must be divisible by nums_kv_head."

        # Each Q head dimension
        self.head_dim = hidden_dim // nums_head
        # Number of Q heads per group
        self.q_heads_per_group = nums_head // nums_kv_head

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_kv_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_kv_head * self.head_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        """
        Forward pass of Group Query Attention.
        
        Args:
            x (Tensor): Input of shape (batch_size, seq_len, hidden_dim).
            attention_mask (BoolTensor, optional): 
                A boolean mask of shape (batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len), 
                where `True` means "allowed to attend" and `False` means "masked out".
                Defaults to None.
        
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim) - the output of attention.
        """
        # 1) Project x to Q, K, V
        batch_size, seq_len, _ = x.size()
        Q = self.q_proj(x)  # (B, seq_len, hidden_dim)
        K = self.k_proj(x)  # (B, seq_len, nums_kv_head * head_dim)
        V = self.v_proj(x)  # (B, seq_len, nums_kv_head * head_dim)

        # 2) Reshape Q
        # Q shape: (B, seq_len, hidden_dim)
        # -> (B, seq_len, nums_head, head_dim)
        # -> (B, nums_head, seq_len, head_dim)
        # -> (B, nums_kv_head, q_heads_per_group, seq_len, head_dim)
        q = Q.view(batch_size, seq_len, self.nums_head, self.head_dim) \
             .transpose(1, 2) \
             .view(batch_size, self.nums_kv_head, self.q_heads_per_group, seq_len, self.head_dim)

        # 3) Reshape K, V
        # K shape: (B, seq_len, nums_kv_head * head_dim)
        # -> (B, seq_len, nums_kv_head, head_dim)
        # -> (B, nums_kv_head, seq_len, head_dim)
        # -> (B, nums_kv_head, 1, seq_len, head_dim)
        k = K.view(batch_size, seq_len, self.nums_kv_head, self.head_dim).transpose(1, 2).unsqueeze(2)
        v = V.view(batch_size, seq_len, self.nums_kv_head, self.head_dim).transpose(1, 2).unsqueeze(2)

        # 4) Compute attention logits
        # (B, nums_kv_head, q_heads_per_group, seq_len, head_dim) @
        # (B, nums_kv_head, 1, head_dim, seq_len)
        # -> (B, nums_kv_head, q_heads_per_group, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # 5) Apply attention mask if provided
        # Note: attention_mask == True means "keep", so we want ~mask for inf fill if mask==False
        if attention_mask is not None:
            # Fill those positions where mask is False with -inf
            attention_scores = attention_scores.masked_fill(~attention_mask, float("-inf"))

        # 6) Apply softmax along the last dimension
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 7) Apply dropout
        attention_weights = self.dropout(attention_weights)

        # 8) Compute the final attention output
        # (B, nums_kv_head, q_heads_per_group, seq_len, seq_len) @
        # (B, nums_kv_head, 1, seq_len, head_dim)
        # -> (B, nums_kv_head, q_heads_per_group, seq_len, head_dim)
        output_tmp = torch.matmul(attention_weights, v)

        # 9) Reshape back to (B, seq_len, hidden_dim)
        # (B, nums_kv_head, q_heads_per_group, seq_len, head_dim)
        # -> (B, nums_head, seq_len, head_dim)
        # -> (B, seq_len, nums_head, head_dim)
        # -> (B, seq_len, hidden_dim)
        output_tmp = output_tmp.view(batch_size, self.nums_head, seq_len, self.head_dim)
        output_concat = output_tmp.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # 10) Final linear projection
        output = self.output_proj(output_concat)
        return output


# -------------------
# Benchmark Function
# -------------------
def benchmark(model, x, n_warmup=5, n_iters=50, attention_mask=None, device='cuda'):
    """
    Run multiple forward passes to measure average latency (in ms) and peak memory usage (in MB).
    In multi-GPU environments, we track the total peak memory across all devices.
    
    Args:
        model (nn.Module): The model to benchmark.
        x (Tensor): Input data of shape (batch_size, seq_len, hidden_dim).
        n_warmup (int): Number of warmup iterations (not timed).
        n_iters (int): Number of timed iterations.
        attention_mask (BoolTensor, optional): A boolean mask for attention.
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        (float, float): Tuple of (average latency in ms, peak memory in MB).
    """
    model.to(device)
    x = x.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Use eval mode to disable dropout
    model.eval()

    # DataParallel only if more than one GPU is available
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        # Reset the peak memory stats
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device=d)
        torch.cuda.empty_cache()

    # Warmup (not timed)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, attention_mask=attention_mask)

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    # Timed iterations
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(x, attention_mask=attention_mask)

    # Synchronize after timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    # Compute average latency
    elapsed = (t1 - t0) / n_iters * 1000.0  # ms

    # Get peak memory
    if torch.cuda.is_available():
        mem_peak = sum(torch.cuda.max_memory_allocated(device=d) for d in range(torch.cuda.device_count()))
        mem_peak = mem_peak / (1024**2)  # in MB
    else:
        mem_peak = 0

    return elapsed, mem_peak


# -------------------
# Main Function
# -------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Example model configurations
    models_config = [
        {
            'model_size': 'Llama3 8B',  # Editable model name
            'layers': 32,
            'hidden_dim': 4096,
            'ffn_dim': 6144,
            'nums_head': 32,
            'nums_kv_head': 8,
            'peak_lr': 3e-4
        }
    ]
    
    # Benchmark settings
    batch_sizes = [16]
    # seq_lens = [128, 256, 512]  # Different sequence lengths
    seq_lens = [512, 1024, 1536]
    n_runs = 5                  # Repeat benchmark for each config
    
    # Different attention configurations
    attention_methods = [
        {
            'method': 'MHA',
            'nums_kv_head': 32,  # nums_kv_head == nums_head
        },
        {
            'method': 'MQA',
            'nums_kv_head': 1,   # nums_kv_head == 1
        },
        {
            'method': 'GQA-8',
            'nums_kv_head': 8,   # generic GQA
        }
    ]

    # List to store results
    results = []

    # Loop over model configs
    for model_conf in models_config:
        model_size = model_conf['model_size']
        hidden_dim = model_conf['hidden_dim']
        nums_head = model_conf['nums_head']
        nums_kv_head = model_conf['nums_kv_head']
        
        print(f"Benchmarking Model Size: {model_size}, Hidden Dim: {hidden_dim}, "
              f"Heads: {nums_head}, KV Heads: {nums_kv_head}")

        # Loop over batch sizes
        for batch_size in batch_sizes:
            # Loop over sequence lengths
            for seq_len in seq_lens:
                # Loop over different attention methods
                for method_conf in attention_methods:
                    method = method_conf['method']
                    current_nums_kv_head = method_conf['nums_kv_head']
                    q_heads_per_group = nums_head // current_nums_kv_head

                    # Initialize the attention layer once per config
                    model = GroupQueryAttention(
                        hidden_dim=hidden_dim,
                        nums_head=nums_head,
                        nums_kv_head=current_nums_kv_head
                    )

                    for run in range(n_runs):
                        # Prepare input data
                        x = torch.randn(batch_size, seq_len, hidden_dim)

                        # Create attention mask of shape (batch_size, nums_kv_head, q_heads_per_group, seq_len, seq_len)
                        attention_mask = torch.tril(torch.ones(
                            batch_size, current_nums_kv_head, q_heads_per_group, seq_len, seq_len
                        )).bool()

                        # Run benchmark
                        elapsed, mem_peak = benchmark(
                            model, x, attention_mask=attention_mask, device=device
                        )

                        # Store results
                        results.append({
                            'Model Size': model_size,
                            'Method': method,
                            'nums_kv_head': current_nums_kv_head,
                            'Seq Length': seq_len,
                            'Time_mean': elapsed,       # in ms
                            'Peak_Mem_mean': mem_peak   # in MB
                        })
                        print(f"Run {run+1}/{n_runs}: Method={method}, "
                              f"nums_kv_head={current_nums_kv_head}, Seq={seq_len}, "
                              f"Time={elapsed:.3f} ms, Peak Mem={mem_peak:.2f} MB")

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Compute mean across runs
    df_stats = df_results.groupby(
        ['Model Size', 'Method', 'nums_kv_head', 'Seq Length']
    ).agg(
        Time_mean=('Time_mean', 'mean'),
        Peak_Mem_mean=('Peak_Mem_mean', 'mean')
    ).reset_index()

    # Sort by Seq Length
    df_stats = df_stats.sort_values(by=['Seq Length'])

    # Reorder columns
    df_stats = df_stats[
        ['Model Size', 'Method', 'nums_kv_head', 'Seq Length', 'Time_mean', 'Peak_Mem_mean']
    ]

    # Print the results table
    print("\n==== Benchmark Results ====\n")
    print(df_stats.to_string(index=False))

    # Save to CSV
    df_stats.to_csv('benchmark_results.csv', index=False)

    # -------------------
    # Visualization
    # -------------------
    sns.set(style="whitegrid")

    # Define the hue order for consistent coloring
    hue_order = ['MHA', 'MQA', 'GQA-8']

    # Plot for each model size
    for model_size in df_stats['Model Size'].unique():
        df_model = df_stats[df_stats['Model Size'] == model_size]

        # ---- Plot average time (ms) ----
        plt.figure(figsize=(12, 8))
        ax_time = sns.barplot(
            data=df_model,
            x='Seq Length',
            y='Time_mean',
            hue='Method',
            palette='viridis',
            edgecolor='black',
            hue_order=hue_order
        )
        ax_time.set_title(f'Average Time (ms) by Method and Seq Length for {model_size} Model', fontsize=16)
        ax_time.set_xlabel('Sequence Length', fontsize=14)
        ax_time.set_ylabel('Time (ms)', fontsize=14)
        plt.legend(title='Method', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        # Annotate the bars with the exact value
        for container in ax_time.containers:
            ax_time.bar_label(container, fmt='%.2f', label_type='edge', padding=3)
        plt.savefig(f'benchmark_time_{model_size.replace(" ", "_")}.svg')
        plt.show()

        # ---- Plot peak memory (MB) ----
        plt.figure(figsize=(12, 8))
        ax_mem = sns.barplot(
            data=df_model,
            x='Seq Length',
            y='Peak_Mem_mean',
            hue='Method',
            palette='magma',
            edgecolor='black',
            hue_order=hue_order
        )
        ax_mem.set_title(f'Average Peak Memory (MB) by Method and Seq Length for {model_size} Model', fontsize=16)
        ax_mem.set_xlabel('Sequence Length', fontsize=14)
        ax_mem.set_ylabel('Peak Memory (MB)', fontsize=14)
        plt.legend(title='Method', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        # Annotate the bars with the exact value
        for container in ax_mem.containers:
            ax_mem.bar_label(container, fmt='%.2f', label_type='edge', padding=3)
        plt.savefig(f'benchmark_mem_{model_size.replace(" ", "_")}.svg')
        plt.show()

    print("\nBenchmarking completed. Results saved to 'benchmark_results.csv' and SVG files.")


if __name__ == "__main__":
    main()
