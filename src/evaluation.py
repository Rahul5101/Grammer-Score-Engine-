import os
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr
import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from .config import CONFIG, SEED

def evaluate_models_and_fit_full(X_train_h, y_train, emb_train, stacker, oof_ensemble_clipped):
    """Refits ensemble on ALL training data to compute compulsory Training RMSE and OOF CV RMSE."""
    full_imputer = SimpleImputer(strategy='median')
    X_full_h_imp = full_imputer.fit_transform(X_train_h)
    full_scaler = StandardScaler()
    X_full_h_scaled = full_scaler.fit_transform(X_full_h_imp)
    full_pca = PCA(n_components=16, random_state=SEED)
    emb_full_pca = full_pca.fit_transform(emb_train)
    X_full_combined = np.hstack([X_full_h_scaled, emb_full_pca])

    full_ridge = Ridge(alpha=10.0).fit(X_full_combined, y_train)
    full_lgb = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=7, random_state=SEED, verbose=-1).fit(X_full_combined, y_train)
    full_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=SEED, verbosity=0).fit(X_full_combined, y_train)
    full_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1).fit(X_full_combined, y_train)

    full_base_preds = np.column_stack([
        full_ridge.predict(X_full_combined),
        full_lgb.predict(X_full_combined),
        full_xgb.predict(X_full_combined),
        full_svr.predict(X_full_combined)
    ])

    train_fit_preds = np.clip(stacker.predict(full_base_preds), 0.0, 5.0)

    train_rmse = math.sqrt(mean_squared_error(y_train, train_fit_preds))
    oof_rmse = math.sqrt(mean_squared_error(y_train, oof_ensemble_clipped))
    oof_mae = mean_absolute_error(y_train, oof_ensemble_clipped)
    oof_pearson, _ = pearsonr(y_train, oof_ensemble_clipped)
    oof_spearman, _ = spearmanr(y_train, oof_ensemble_clipped)
    oof_r2 = r2_score(y_train, oof_ensemble_clipped)
    oof_qwk = cohen_kappa_score(np.round(y_train).astype(int), np.round(oof_ensemble_clipped).astype(int), weights='quadratic')

    print("="*60)
    print(f"*** COMPULSORY TRAINING RMSE (Full Train Refit): {train_rmse:.4f} ***")
    print(f"*** OUT-OF-FOLD CV RMSE (Honest Estimate):     {oof_rmse:.4f} ***")
    print(f"OOF MAE: {oof_mae:.4f} | Pearson r: {oof_pearson:.4f} | Spearman rho: {oof_spearman:.4f} | R2: {oof_r2:.4f} | QWK: {oof_qwk:.4f}")
    print("="*60)

    # True vs Predicted Scatter
    plt.figure(figsize=(7, 7))
    plt.scatter(y_train, oof_ensemble_clipped, alpha=0.5, color='darkblue', edgecolors='k', label='OOF Predictions')
    plt.plot([0, 5], [0, 5], 'r--', linewidth=2, label='Perfect Fit (y=x)')
    plt.title(f"Predicted vs True Scores (OOF RMSE: {oof_rmse:.4f})", fontsize=14, fontweight='bold')
    plt.xlabel("True Grammar Score")
    plt.ylabel("Predicted Grammar Score")
    plt.xlim(-0.2, 5.2)
    plt.ylim(-0.2, 5.2)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['figures_dir'], "true_vs_pred.png"), dpi=300)
    plt.close()

    # Residuals Plot
    residuals = oof_ensemble_clipped - y_train
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='crimson', bins=15)
    plt.title("Residuals Distribution (Predicted - True)", fontsize=14, fontweight='bold')
    plt.xlabel("Residual Error")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['figures_dir'], "residuals.png"), dpi=300)
    plt.close()

    metrics_summary = {
        'train_rmse': train_rmse,
        'oof_rmse': oof_rmse,
        'oof_mae': oof_mae,
        'oof_pearson': oof_pearson,
        'oof_spearman': oof_spearman,
        'oof_r2': oof_r2,
        'oof_qwk': oof_qwk
    }

    full_models = {
        'ridge': full_ridge,
        'lgb': full_lgb,
        'xgb': full_xgb,
        'svr': full_svr,
        'X_full_combined': X_full_combined
    }

    return metrics_summary, full_models
