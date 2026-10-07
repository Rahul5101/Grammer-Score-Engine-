import os
import re
import json
import time
import numpy as np
import pandas as pd
import librosa
import whisper
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from .config import CONFIG

def load_and_preprocess_audio(audio_path, sr=16000):
    """Loads audio at 16kHz mono, applies peak normalization, trims silence (top_db=30)."""
    y, sr_out = librosa.load(audio_path, sr=sr, mono=True)
    y_norm = librosa.util.normalize(y)
    raw_dur = float(len(y_norm) / sr_out)
    y_trimmed, _ = librosa.effects.trim(y_norm, top_db=30)
    trimmed_dur = float(len(y_trimmed) / sr_out)
    return y_norm, y_trimmed, raw_dur, trimmed_dur

def run_asr_pipeline(train_df, test_df):
    """Runs Whisper ASR model on audio files and caches transcripts with confidence scores."""
    cache_path = os.path.join(CONFIG['cache_dir'], "transcripts.parquet")

    expected_keys = set(('train', f) for f in train_df['filename']).union(set(('test', f) for f in test_df['filename']))
    if os.path.exists(cache_path):
        transcripts_df = pd.read_parquet(cache_path)
        transcripts_df = transcripts_df.drop_duplicates(subset=['split', 'filename']).reset_index(drop=True)
        cached_keys = set(zip(transcripts_df['split'], transcripts_df['filename']))
        if len(transcripts_df) >= len(expected_keys) and expected_keys.issubset(cached_keys):
            print("Loading transcripts from parquet cache...", flush=True)
            return transcripts_df

    print("Loading Whisper model for ASR transcription...", flush=True)
    whisper_model = whisper.load_model("tiny.en", device=CONFIG['device'])
    
    all_transcripts = []
    if os.path.exists(cache_path):
        existing_df = pd.read_parquet(cache_path)
        all_transcripts = existing_df.to_dict('records')
        processed_files = set((r['split'], r['filename']) for r in all_transcripts)
    else:
        processed_files = set()

    tasks = []
    for idx, row in train_df.iterrows():
        if ('train', row['filename']) not in processed_files:
            tasks.append((row['filename'], row['label'], 'train'))
    for idx, row in test_df.iterrows():
        if ('test', row['filename']) not in processed_files:
            tasks.append((row['filename'], row.get('label', np.nan), 'test'))

    print(f"Transcribing {len(tasks)} remaining audio files with Whisper...", flush=True)
    t0 = time.time()

    for idx, (fname, lbl, split_name) in enumerate(tasks):
        audio_dir = os.path.join(CONFIG['data_dir'], split_name)
        audio_path = os.path.join(audio_dir, fname)
        try:
            y_norm, y_trimmed, r_dur, t_dur = load_and_preprocess_audio(audio_path)
            res = whisper_model.transcribe(y_norm, language="en", word_timestamps=True, temperature=0, condition_on_previous_text=False)
            
            raw_text = res.get('text', '').strip()
            clean_text = re.sub(r'\s+', ' ', raw_text.lower()).strip()
            
            words_info = []
            avg_logprobs = []
            no_speech_probs = []
            for seg in res.get('segments', []):
                if 'avg_logprob' in seg:
                    avg_logprobs.append(seg['avg_logprob'])
                if 'no_speech_prob' in seg:
                    no_speech_probs.append(seg['no_speech_prob'])
                if 'words' in seg:
                    for w in seg['words']:
                        words_info.append({
                            'word': w.get('word', ''),
                            'start': float(w.get('start', 0.0)),
                            'end': float(w.get('end', 0.0)),
                            'probability': float(w.get('probability', 0.0))
                        })
            
            word_probs = [w['probability'] for w in words_info if 'probability' in w]
            mean_w_prob = float(np.mean(word_probs)) if len(word_probs) > 0 else 0.0
            min_w_prob = float(np.min(word_probs)) if len(word_probs) > 0 else 0.0
            low_conf_ratio = float(np.mean([p < 0.5 for p in word_probs])) if len(word_probs) > 0 else 0.0
            avg_logprob = float(np.mean(avg_logprobs)) if len(avg_logprobs) > 0 else 0.0
            no_speech_prob = float(np.mean(no_speech_probs)) if len(no_speech_probs) > 0 else 0.0
            
            all_transcripts.append({
                'filename': fname,
                'split': split_name,
                'label': lbl,
                'raw_dur': r_dur,
                'trimmed_dur': t_dur,
                'raw_transcript': raw_text,
                'clean_transcript': clean_text,
                'mean_word_prob': mean_w_prob,
                'min_word_prob': min_w_prob,
                'low_conf_ratio': low_conf_ratio,
                'whisper_avg_logprob': avg_logprob,
                'whisper_no_speech_prob': no_speech_prob,
                'words_json': json.dumps(words_info)
            })
        except Exception as e:
            print(f"Error processing {split_name}/{fname}: {e}", flush=True)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(tasks):
            elapsed = time.time() - t0
            print(f"ASR Progress: {idx+1}/{len(tasks)} transcribed ({elapsed:.1f}s elapsed, {(idx+1)/max(elapsed, 0.1):.2f} samples/s)", flush=True)
            df_temp = pd.DataFrame(all_transcripts).drop_duplicates(subset=['split', 'filename']).reset_index(drop=True)
            df_temp.to_parquet(cache_path)

    transcripts_df = pd.DataFrame(all_transcripts).drop_duplicates(subset=['split', 'filename']).reset_index(drop=True)
    transcripts_df.to_parquet(cache_path)

    # Duration Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(transcripts_df['trimmed_dur'], kde=True, color='teal', bins=15)
    plt.title("Audio Duration Histogram (Post Silence Trimming)", fontsize=14, fontweight='bold')
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['figures_dir'], "duration_hist.png"), dpi=300)
    plt.close()

    return transcripts_df
