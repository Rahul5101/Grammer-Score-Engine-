import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from .config import CONFIG

def load_data():
    """Loads train.csv, test.csv, and sample_submission.csv."""
    train_path = os.path.join(CONFIG['data_dir'], "train.csv")
    test_path = os.path.join(CONFIG['data_dir'], "test.csv")
    sub_path = os.path.join(CONFIG['data_dir'], "sample_submission.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub_df = pd.read_csv(sub_path)

    return train_df, test_df, sample_sub_df

def run_eda(train_df):
    """Generates EDA distribution plots for target label."""
    plt.figure(figsize=(8, 5))
    sns.histplot(train_df['label'], kde=True, bins=10, color='indigo')
    plt.title("Train Label Distribution (Grammar Scores 0 - 5)", fontsize=14, fontweight='bold')
    plt.xlabel("Grammar Score")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(CONFIG['figures_dir'], "label_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path
