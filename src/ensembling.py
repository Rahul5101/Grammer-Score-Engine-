import os
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import CONFIG

def fit_stacker_and_ensemble(oof_dict, test_dict, y_train):
    """Fits non-negative Ridge stacker over OOF model predictions and clips outputs to [0, 5]."""
    OOF_matrix = np.column_stack([oof_dict['ridge'], oof_dict['lgb'], oof_dict['xgb'], oof_dict['svr']])
    TEST_matrix = np.column_stack([test_dict['ridge'], test_dict['lgb'], test_dict['xgb'], test_dict['svr']])

    stacker = Ridge(alpha=1.0, positive=True)
    stacker.fit(OOF_matrix, y_train)

    oof_ensemble = stacker.predict(OOF_matrix)
    oof_ensemble_clipped = np.clip(oof_ensemble, 0.0, 5.0)

    test_ensemble = stacker.predict(TEST_matrix)
    test_ensemble_clipped = np.clip(test_ensemble, 0.0, 5.0)

    return stacker, oof_ensemble_clipped, test_ensemble_clipped

def run_ablation_study(df_train, feature_cols, feats_text_df, feats_audio_df, emb_text_train, y_train, oof_rmse_val):
    """Runs ablation study comparing Text-only, Acoustic-only, Embeddings-only, and Full Multi-View Stack."""
    text_only_cols = [c for c in feature_cols if c in feats_text_df.columns]
    X_text_imp = SimpleImputer(strategy='median').fit_transform(df_train[text_only_cols].values)
    r_text = Ridge(alpha=10.0).fit(StandardScaler().fit_transform(X_text_imp), y_train)
    rmse_text = math.sqrt(mean_squared_error(y_train, np.clip(r_text.predict(StandardScaler().fit_transform(X_text_imp)), 0, 5)))

    ac_only_cols = [c for c in feature_cols if c in feats_audio_df.columns]
    X_ac_imp = SimpleImputer(strategy='median').fit_transform(df_train[ac_only_cols].values)
    r_ac = Ridge(alpha=10.0).fit(StandardScaler().fit_transform(X_ac_imp), y_train)
    rmse_ac = math.sqrt(mean_squared_error(y_train, np.clip(r_ac.predict(StandardScaler().fit_transform(X_ac_imp)), 0, 5)))

    r_emb = Ridge(alpha=10.0).fit(emb_text_train, y_train)
    rmse_emb = math.sqrt(mean_squared_error(y_train, np.clip(r_emb.predict(emb_text_train), 0, 5)))

    ablation_labels = ['Acoustic-Only', 'Text-Only', 'Embeddings-Only', 'Full Multi-View Stack']
    ablation_rmses = [rmse_ac, rmse_text, rmse_emb, oof_rmse_val]

    plt.figure(figsize=(8, 5))
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    bars = plt.bar(ablation_labels, ablation_rmses, color=colors, width=0.5, edgecolor='black')
    plt.title("Ablation Study: RMSE Across Feature Views", fontsize=14, fontweight='bold')
    plt.ylabel("RMSE (Lower is Better)")
    plt.ylim(0, max(ablation_rmses) * 1.2)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['figures_dir'], "ablation_chart.png"), dpi=300)
    plt.close()

    return pd.DataFrame({'View': ablation_labels, 'OOF_RMSE': ablation_rmses})
