import os
import numpy as np
import pandas as pd
from aeon.datasets import load_classification

# Path where your datasets are stored
data_path = "./home/yanivgra/dinov3/dinov3/data/multivariate"

# List to store our results
summary_data = []

# Get all folder names in the directory
dataset_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

print(f"Found {len(dataset_folders)} datasets. Analyzing...")

for name in dataset_folders:
    try:
        # Load the training set to get metadata
        X, y = load_classification(name, extract_path=data_path, split="train")
        
        # X shape is (n_instances, n_channels, n_timepoints)
        n_samples, n_channels, n_length = X.shape

        # Per-class (category) counts within this dataset
        classes, counts = np.unique(y, return_counts=True)
        class_counts = {str(c): int(n) for c, n in zip(classes, counts)}
        n_classes = len(classes)

        summary_data.append({
            "Dataset Name": name,
            "Samples (Train)": n_samples,
            "Channels": n_channels,
            "Sequence Length": n_length,
            "Number of Classes": n_classes,
            "Total Shards (Samples * Channels)": n_samples * n_channels,
            "Samples per Class": class_counts,
        })

        # Console summary for this dataset
        print(f"✅ Processed {name}")
        print(f"   Total samples: {n_samples}")
        print(f"   Classes ({n_classes}): {class_counts}")
        
    except Exception as e:
        print(f"❌ Could not process {name}: {e}")

# Create a DataFrame and save to CSV
df = pd.DataFrame(summary_data)
df = df.sort_values(by="Total Shards (Samples * Channels)", ascending=False)
df.to_csv("ucr_multivariate_summary.csv", index=False)

# Also save as an HTML page with a table
df.to_html("ucr_multivariate_summary.html", index=False)

print("\n--- Summary Report ---")
print(df.to_string(index=False))

total_multivariate_samples = int(df["Samples (Train)"].sum())
total_channels = int(df["Channels"].sum())
total_univariate_samples = int(df["Total Shards (Samples * Channels)"].sum())

print("\n--- Global Totals ---")
print(f"Total multivariate samples (original instances): {total_multivariate_samples}")
print(f"Total channels across all datasets: {total_channels}")
print(f"Total univariate samples after channel split (Samples * Channels): {total_univariate_samples}")

print("\nSummary saved to 'ucr_multivariate_summary.csv' and 'ucr_multivariate_summary.html'")