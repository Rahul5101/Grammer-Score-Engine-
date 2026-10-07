import os
import json
import numpy as np
import pandas as pd
import librosa
from .config import CONFIG

def extract_audio_features(transcripts_df):
    """Extracts acoustic features (librosa MFCCs, RMS, ZCR, centroid) and speech timing (WPM, pause stats)."""
    cache_path = os.path.join(CONFIG['cache_dir'], "feats_audio.parquet")
    
    if os.path.exists(cache_path):
        feats_df = pd.read_parquet(cache_path)
        if 'split' in feats_df.columns:
            feats_df = feats_df.drop_duplicates(subset=['split', 'filename']).reset_index(drop=True)
            cached_keys = set(zip(feats_df['split'], feats_df['filename']))
            req_keys = set(zip(transcripts_df['split'], transcripts_df['filename']))
            if len(feats_df) == len(transcripts_df) and cached_keys == req_keys:
                print("Loading cached acoustic features...")
                return feats_df
        else:
            print("Cached acoustic features missing 'split' column. Recomputing...")
        print(f"Cached acoustic features mismatch or incomplete ({len(feats_df)} vs {len(transcripts_df)}). Recomputing...")

    print("Extracting Acoustic and Speech Timing Features...")
    audio_feat_list = []

    for idx, row in transcripts_df.iterrows():
        fname = row['filename']
        split_name = row['split']
        clean_text = row['clean_transcript']
        trimmed_dur = max(row['trimmed_dur'], 1.0)
        words = clean_text.split()
        num_words = max(len(words), 1)

        words_data = json.loads(row['words_json'])
        speech_dur = 0.0
        pauses_03 = 0
        pauses_05 = 0
        pause_lens = []

        if len(words_data) > 1:
            for i in range(len(words_data) - 1):
                w_curr = words_data[i]
                w_next = words_data[i+1]
                p_len = w_next['start'] - w_curr['end']
                if p_len > 0.3:
                    pauses_03 += 1
                    pause_lens.append(p_len)
                if p_len > 0.5:
                    pauses_05 += 1
                speech_dur += max(w_curr['end'] - w_curr['start'], 0.0)
            speech_dur += max(words_data[-1]['end'] - words_data[-1]['start'], 0.0)
        else:
            speech_dur = trimmed_dur * 0.8

        wpm = (num_words / trimmed_dur) * 60.0
        articulation_rate = num_words / max(speech_dur, 0.1)
        mean_pause_len = float(np.mean(pause_lens)) if len(pause_lens) > 0 else 0.0
        max_pause_len = float(np.max(pause_lens)) if len(pause_lens) > 0 else 0.0
        speech_time_ratio = speech_dur / trimmed_dur

        audio_path = os.path.join(CONFIG['data_dir'], split_name, fname)
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            y_norm = librosa.util.normalize(y)
            mfccs = librosa.feature.mfcc(y=y_norm, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_stds = np.std(mfccs, axis=1)

            rms = librosa.feature.rms(y=y_norm)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))

            zcr = librosa.feature.zero_crossing_rate(y=y_norm)[0]
            zcr_mean = float(np.mean(zcr))
            zcr_std = float(np.std(zcr))

            centroid = librosa.feature.spectral_centroid(y=y_norm, sr=sr)[0]
            centroid_mean = float(np.mean(centroid))
            centroid_std = float(np.std(centroid))
        except Exception:
            mfcc_means = np.zeros(13)
            mfcc_stds = np.zeros(13)
            rms_mean, rms_std = 0.0, 0.0
            zcr_mean, zcr_std = 0.0, 0.0
            centroid_mean, centroid_std = 0.0, 0.0

        a_feat = {
            'filename': fname,
            'split': split_name,
            'wpm': wpm,
            'articulation_rate': articulation_rate,
            'pause_count_03': pauses_03,
            'pause_count_05': pauses_05,
            'mean_pause_len': mean_pause_len,
            'max_pause_len': max_pause_len,
            'speech_time_ratio': speech_time_ratio,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'centroid_mean': centroid_mean,
            'centroid_std': centroid_std,
            'whisper_mean_word_prob': row['mean_word_prob'],
            'whisper_min_word_prob': row['min_word_prob'],
            'whisper_low_conf_ratio': row['low_conf_ratio'],
            'whisper_avg_logprob': row['whisper_avg_logprob'],
            'whisper_no_speech_prob': row['whisper_no_speech_prob']
        }
        for m_i in range(13):
            a_feat[f'mfcc_mean_{m_i}'] = float(mfcc_means[m_i])
            a_feat[f'mfcc_std_{m_i}'] = float(mfcc_stds[m_i])

        audio_feat_list.append(a_feat)

    feats_audio_df = pd.DataFrame(audio_feat_list).drop_duplicates(subset=['split', 'filename']).reset_index(drop=True)
    feats_audio_df.to_parquet(cache_path)
    return feats_audio_df
