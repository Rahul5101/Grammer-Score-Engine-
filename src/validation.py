import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
from .config import SEED

def run_cross_validation(X_train_h, y_train, emb_train, X_test_h, emb_test, n_splits=5, n_repeats=2):
    """Executes 5-Fold Repeated Stratified CV over handcrafted features and embeddings."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    strat_y = np.round(y_train).astype(int)

    oof_baseline = np.zeros(len(y_train))
    oof_ridge = np.zeros(len(y_train))
    oof_lgb = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))
    oof_svr = np.zeros(len(y_train))
 
    test_preds_ridge = np.zeros(len(X_test_h))
    test_preds_lgb = np.zeros(len(X_test_h))
    test_preds_xgb = np.zeros(len(X_test_h))
    test_preds_svr = np.zeros(len(X_test_h))

    print("Running Repeated 5-Fold Cross Validation...")

    for fold, (train_idx, val_idx) in enumerate(rskf.split(X_train_h, strat_y)):
        X_tr_h, y_tr = X_train_h[train_idx], y_train[train_idx]
        X_va_h, y_va = X_train_h[val_idx], y_train[val_idx]
        
        emb_tr, emb_va = emb_train[train_idx], emb_train[val_idx]
        
        # In-fold leakage-free preprocessing
        imputer = SimpleImputer(strategy='median')
        X_tr_h = imputer.fit_transform(X_tr_h)
        X_va_h = imputer.transform(X_va_h)
        
        scaler = StandardScaler()
        X_tr_h_scaled = scaler.fit_transform(X_tr_h)
        X_va_h_scaled = scaler.transform(X_va_h)
        
        pca = PCA(n_components=16, random_state=SEED)
        emb_tr_pca = pca.fit_transform(emb_tr)
        emb_va_pca = pca.transform(emb_va)
        
        X_tr_combined = np.hstack([X_tr_h_scaled, emb_tr_pca])
        X_va_combined = np.hstack([X_va_h_scaled, emb_va_pca])
        
        # Baseline: Predict Train Mean
        mean_val = np.mean(y_tr)
        oof_baseline[val_idx] += mean_val / n_repeats
        
        # Ridge
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr_combined, y_tr)
        oof_ridge[val_idx] += ridge.predict(X_va_combined) / n_repeats
        
        X_te_h_imp = imputer.transform(X_test_h)
        X_te_h_scaled = scaler.transform(X_te_h_imp)
        emb_te_pca = pca.transform(emb_test)
        X_te_combined = np.hstack([X_te_h_scaled, emb_te_pca])
        test_preds_ridge += ridge.predict(X_te_combined) / (n_splits * n_repeats)
        
        # LightGBM
        lgbm = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=7, random_state=SEED, verbose=-1)
        lgbm.fit(X_tr_combined, y_tr)
        oof_lgb[val_idx] += lgbm.predict(X_va_combined) / n_repeats
        test_preds_lgb += lgbm.predict(X_te_combined) / (n_splits * n_repeats)
        
        # XGBoost
        xgbr = xgb.XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=SEED, verbosity=0)
        xgbr.fit(X_tr_combined, y_tr)
        oof_xgb[val_idx] += xgbr.predict(X_va_combined) / n_repeats
        test_preds_xgb += xgbr.predict(X_te_combined) / (n_splits * n_repeats)
        
        # SVR
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr.fit(X_tr_combined, y_tr)
        oof_svr[val_idx] += svr.predict(X_va_combined) / n_repeats
        test_preds_svr += svr.predict(X_te_combined) / (n_splits * n_repeats)

    oof_dict = {
        'baseline': oof_baseline,
        'ridge': oof_ridge,
        'lgb': oof_lgb,
        'xgb': oof_xgb,
        'svr': oof_svr
    }
    
    test_dict = {
        'ridge': test_preds_ridge,
        'lgb': test_preds_lgb,
        'xgb': test_preds_xgb,
        'svr': test_preds_svr
    }
    
    return oof_dict, test_dict
