"""
Main entrypoint for Grammar Scoring Engine Pipeline.
Integrates all modules from src/ to run end-to-end training, cross-validation, evaluation, and test inference.
"""

import os
import pandas as pd
from src.config import CONFIG, SEED, seed_everything
from src.eda import load_data, run_eda
from src.asr import run_asr_pipeline
from src.features_text import extract_text_features
from src.features_audio import extract_audio_features
from src.embeddings import extract_text_embeddings
from src.validation import run_cross_validation
from src.ensembling import fit_stacker_and_ensemble, run_ablation_study
from src.evaluation import evaluate_models_and_fit_full
from src.interpretability import run_interpretability
from src.inference import generate_submission

def main():
    print("="*70)
    print("      GRAMMAR SCORING ENGINE - MULTI-VIEW PIPELINE      ")
    print("="*70)
    seed_everything(SEED)

    # 1. Load Data & Run EDA
    print("\n[Step 1] Data Loading & Metadata Audit...")
    train_df, test_df, sample_sub_df = load_data()
    run_eda(train_df)

    # 2. ASR Transcription
    print("\n[Step 2] ASR Transcription with Whisper...")
    transcripts_df = run_asr_pipeline(train_df, test_df)

    # 3. Feature Extraction
    print("\n[Step 3] Multi-View Feature Groups Extraction...")
    feats_text_df = extract_text_features(transcripts_df)
    feats_audio_df = extract_audio_features(transcripts_df)
    emb_text = extract_text_embeddings(transcripts_df)

    # Merge Feature Matrices
    merged_df = transcripts_df[['filename', 'split', 'label']].copy()
    merged_df = merged_df.merge(feats_text_df, on=['split', 'filename'], how='left')
    merged_df = merged_df.merge(feats_audio_df, on=['split', 'filename'], how='left')

    train_mask = merged_df['split'] == 'train'
    test_mask = merged_df['split'] == 'test'

    df_train = merged_df[train_mask].reset_index(drop=True)
    df_test = merged_df[test_mask].reset_index(drop=True)

    feature_cols = [c for c in df_train.columns if c not in ['filename', 'split', 'label']]
    X_train_h = df_train[feature_cols].values
    y_train = df_train['label'].values
    X_test_h = df_test[feature_cols].values

    emb_train = emb_text[train_mask]
    emb_test = emb_text[test_mask]

    # 4. Cross Validation
    print("\n[Step 4] 5-Fold Repeated Stratified Cross-Validation...")
    oof_dict, test_dict = run_cross_validation(X_train_h, y_train, emb_train, X_test_h, emb_test)

    # 5. Multi-view Stacking & Ensembling
    print("\n[Step 5] Multi-View Stacking & Ensembling...")
    stacker, oof_ensemble_clipped, test_ensemble_clipped = fit_stacker_and_ensemble(oof_dict, test_dict, y_train)

    # 6. Evaluation & Compulsory Training RMSE
    print("\n[Step 6] Compulsory Metric Evaluation & Full Training Set Refit...")
    metrics, full_models = evaluate_models_and_fit_full(X_train_h, y_train, emb_train, stacker, oof_ensemble_clipped)

    # 7. Ablation Study
    print("\n[Step 7] Running Feature View Ablation Study...")
    ablation_df = run_ablation_study(df_train, feature_cols, feats_text_df, feats_audio_df, emb_train, y_train, metrics['oof_rmse'])
    print(ablation_df)

    # 8. Interpretability & SHAP
    print("\n[Step 8] Model Interpretability & SHAP Analysis...")
    run_interpretability(full_models, feature_cols)

    # 9. Test Set Inference & Submission Integrity
    print("\n[Step 9] Generating Test Submission...")
    sub_df = generate_submission(df_test, test_ensemble_clipped)
    print("\nPipeline execution complete! Deliverable files ready.")

if __name__ == "__main__":
    main()
