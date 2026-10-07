import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from .config import CONFIG

def run_interpretability(full_models, feature_cols):
    """Generates LightGBM Gain Feature Importances and SHAP summary plot."""
    full_lgb = full_models['lgb']
    X_full_combined = full_models['X_full_combined']

    lgb_feat_imp = pd.Series(full_lgb.feature_importances_[:len(feature_cols)], index=feature_cols).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    lgb_feat_imp.head(12).plot(kind='barh', color='darkcyan', edgecolor='black')
    plt.gca().invert_yaxis()
    plt.title("Top 12 Feature Importances (LightGBM)", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['figures_dir'], "feature_importance.png"), dpi=300)
    plt.close()

    explainer = shap.TreeExplainer(full_lgb)
    shap_values = explainer.shap_values(X_full_combined)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_full_combined, show=False, max_display=10)
    plt.title("SHAP Summary Plot (Multi-View Model)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['figures_dir'], "shap_summary.png"), dpi=300)
    plt.close()

    return lgb_feat_imp
