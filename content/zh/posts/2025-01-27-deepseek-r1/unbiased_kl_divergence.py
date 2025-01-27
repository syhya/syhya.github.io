import torch
import torch.distributions as dis
import numpy as np
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Determine device: use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters for the experiment
loc_values = [0.1, 0.5, 1.0]  # Representative loc values
num_samples = 500_000_000  # Large sample size for accuracy
results_list = []

# Define q distribution (fixed) and move parameters to device
q_loc = torch.tensor(0.0).to(device) # Convert to tensor and move to device
q_scale = torch.tensor(1.0).to(device) # Convert to tensor and move to device
q = dis.Normal(loc=q_loc, scale=q_scale) # Create Normal distribution with device parameters

# Loop through different loc values for p
for loc in loc_values:
    # Define p distribution and move parameters to device
    p_loc = torch.tensor(loc).to(device) # Convert to tensor and move to device
    p_scale = torch.tensor(1.0).to(device) # Convert to tensor and move to device
    p = dis.Normal(loc=p_loc, scale=p_scale) # Create Normal distribution with device parameters
    truekl = dis.kl_divergence(p, q).item()

    # Sample from q (samples will be on the same device as q)
    x = q.sample(sample_shape=(num_samples,)) # Samples are generated on the device of q
    logr = p.log_prob(x) - q.log_prob(x)

    k1 = -logr
    k2 = logr ** 2 / 2
    k3 = (logr.exp() - 1) - logr

    estimators = {'k1': k1, 'k2': k2, 'k3': k3}
    for name, k in estimators.items():
        mean_estimate = k.mean().item()
        std_dev = k.std().item()
        relative_bias = (mean_estimate - truekl) / truekl * 100 if truekl != 0 else mean_estimate * 100 # Relative bias in percentage
        results_list.append({
            'True KL Divergence': f"{truekl:.4f}",
            'Estimator': name,
            'Mean Estimate': f"{mean_estimate:.4f}",
            'Standard Deviation': f"{std_dev:.4f}",
            'Relative Bias (%)': f"{relative_bias:.4f}%" # More decimal places for bias
        })

# Create Pandas DataFrame for table formatting
df = pd.DataFrame(results_list)

# Print Markdown table
markdown_table = df.to_markdown(index=False)
print(markdown_table)