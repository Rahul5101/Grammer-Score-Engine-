import os
import pandas as pd
from .config import CONFIG

def generate_submission(df_test, test_ensemble_clipped):
    """Generates and validates outputs/submission.csv matching sample_submission.csv format."""
    sub_df = pd.DataFrame({
        'filename': df_test['filename'],
        'label': test_ensemble_clipped
    })

    sub_path = os.path.join(CONFIG['outputs_dir'], "submission.csv")
    sub_df.to_csv(sub_path, index=False)

    print(f"Submission file created at {sub_path} with shape {sub_df.shape}")

    # Integrity Assertions
    assert len(sub_df) == 216, f"Expected 216 rows, got {len(sub_df)}"
    assert sub_df['label'].isnull().sum() == 0, "Submission contains NaN values!"
    assert sub_df['label'].min() >= 0.0, "Submission contains negative values!"
    assert sub_df['label'].max() <= 5.0, "Submission contains values > 5.0!"
    assert list(sub_df['filename']) == list(df_test['filename']), "Filename ordering mismatch!"

    print("ALL SUBMISSION INTEGRITY ASSERTIONS PASSED SUCCESSFULLY!")
    return sub_df
