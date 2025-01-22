"""
compare three normalization methods: BatchNorm, LayerNorm, and RMSNorm
"""
import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    """
    Custom Batch Normalization layer.
    
    This class implements a simplified version of batch normalization
    for a 3D input tensor with shape (batch_size, seq_len, hidden_dim).
    The normalization is performed across the batch dimension (dim=0).
    
    Attributes:
        gamma (nn.Parameter): Scale parameter, shape (hidden_dim).
        beta (nn.Parameter): Bias parameter, shape (hidden_dim).
        eps (float): A small value to avoid division by zero.
    """
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps
    
    def forward(self, x):
        """
        Forward pass of the Batch Normalization layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        
        Returns:
            torch.Tensor: Output tensor of the same shape after batch normalization.
        """
        # Compute mean and variance across the batch dimension (dim=0).
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        output = self.gamma * x_norm + self.beta
        return output


class LayerNorm(nn.Module):
    """
    Custom Layer Normalization layer.
    
    This class implements a simplified version of layer normalization
    for a 3D input tensor with shape (batch_size, seq_len, hidden_dim).
    The normalization is performed across the last dimension (dim=-1).
    
    Attributes:
        gamma (nn.Parameter): Scale parameter, shape (hidden_dim).
        beta (nn.Parameter): Bias parameter, shape (hidden_dim).
        eps (float): A small value to avoid division by zero.
    """
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps
  
    def forward(self, x):
        """
        Forward pass of the Layer Normalization layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        
        Returns:
            torch.Tensor: Output tensor of the same shape after layer normalization.
        """
        # Compute mean and variance across the last dimension (hidden_dim).
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        output = self.gamma * x_norm + self.beta
        return output


class RMSNorm(nn.Module):
    """
    Custom RMS Normalization layer.
    
    This class implements a root mean square normalization for a 3D input
    tensor with shape (batch_size, seq_len, hidden_dim). It normalizes
    the tensor by dividing each element by the RMS over the last dimension,
    then multiplies by a learnable scale gamma.
    
    Attributes:
        gamma (nn.Parameter): Scale parameter, shape (hidden_dim).
        eps (float): A small value to avoid division by zero.
    """
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps
    
    def forward(self, x):
        """
        Forward pass of the RMS Normalization layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        
        Returns:
            torch.Tensor: Output tensor of the same shape after RMS normalization.
        """
        # Compute root mean square along the last dimension.
        rmse = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize by RMS
        x_norm = x / rmse

        # Scale by gamma
        output = x_norm * self.gamma
        return output


if __name__ == "__main__":
    torch.manual_seed(42)  # Set the random seed for reproducibility

    # 1. Prepare random input data
    batch_size = 4
    seq_len = 5
    hidden_dim = 6
    x_input = torch.randn(batch_size, seq_len, hidden_dim)
    
    print("Input shape:", x_input.shape)
    print("Input mean:", x_input.mean().item())
    print("Input variance:", x_input.var().item())

    # 2. Test BatchNorm
    batch_norm = BatchNorm(hidden_dim)
    x_batch_norm = batch_norm(x_input)
    print("\n[BatchNorm] Output shape:", x_batch_norm.shape)
    print("[BatchNorm] Output mean (dim=0):", x_batch_norm.mean(dim=0).mean().item())
    print("[BatchNorm] Output variance (dim=0):", x_batch_norm.var(dim=0, unbiased=False).mean().item())

    # 3. Test LayerNorm
    layer_norm = LayerNorm(hidden_dim)
    x_layer_norm = layer_norm(x_input)
    print("\n[LayerNorm] Output shape:", x_layer_norm.shape)
    print("[LayerNorm] Output mean (dim=-1):", x_layer_norm.mean(dim=-1).mean().item())
    print("[LayerNorm] Output variance (dim=-1):", x_layer_norm.var(dim=-1, unbiased=False).mean().item())

    # 4. Test RMSNorm
    rms_norm = RMSNorm(hidden_dim)
    x_rms_norm = rms_norm(x_input)
    print("\n[RMSNorm] Output shape:", x_rms_norm.shape)
    # Calculate Root Mean Square (RMS) for input and output
    rms_input = torch.mean(x_input**2, dim=-1).sqrt().mean().item()
    rms_output = torch.mean(x_rms_norm**2, dim=-1).sqrt().mean().item()
    print(f"[RMSNorm] Input RMS (avg): {rms_input}")
    print(f"[RMSNorm] Output RMS (avg): {rms_output}")
