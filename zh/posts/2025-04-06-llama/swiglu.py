import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNSwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Feedforward neural network layer based on the SwiGLU activation function.
        :param input_dim: Input dimension
        :param hidden_dim: Hidden layer dimension
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)  # Scale according to llama paper recommendation
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)
        
    def forward(self, x):
        """
        Forward pass
        :param x: Input tensor with shape (batch_size, seq_len, input_dim)
        :return: Processed tensor with shape (batch_size, seq_len, input_dim)
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FFNSwiGLUFromScratch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Feedforward neural network layer with manually implemented SwiGLU activation function.
        :param input_dim: Input dimension
        :param hidden_dim: Hidden layer dimension
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)
    
    def sigmoid(self, x):
        """
        Implementation of the Sigmoid activation function
        """
        return 1 / (1 + torch.exp(-x))
    
    def silu(self, x):
        """
        Implementation of the SiLU (Swish) activation function
        """
        return x * self.sigmoid(x)
    
    def forward(self, x):
        """
        Forward pass
        :param x: Input tensor with shape (batch_size, seq_len, input_dim)
        :return: Processed tensor with shape (batch_size, seq_len, input_dim)
        """
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


if __name__ == "__main__":
    batch_size = 128
    seq_len = 256
    input_dim = 2048
    hidden_dim = 4096
    x = torch.randn(batch_size, seq_len, input_dim)
    layer1 = FFNSwiGLU(input_dim, hidden_dim)
    layer2 = FFNSwiGLUFromScratch(input_dim, hidden_dim)
    x_ffn1 = layer1(x)
    x_ffn2 = layer2(x)
    print(x_ffn1.size())
    print(x_ffn2.size())

