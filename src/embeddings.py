import os
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import CONFIG

def extract_text_embeddings(transcripts_df):
    """Computes SentenceTransformers embeddings (all-MiniLM-L6-v2) for transcripts."""
    cache_path = os.path.join(CONFIG['cache_dir'], "emb_text.npy")
    
    if os.path.exists(cache_path):
        emb_text = np.load(cache_path)
        if len(emb_text) == len(transcripts_df):
            print("Loading cached text embeddings...")
            return emb_text
        else:
            print(f"Cached text embeddings count ({len(emb_text)}) does not match transcripts count ({len(transcripts_df)}). Recomputing...")

    print("Computing SentenceTransformers text embeddings...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    raw_texts = transcripts_df['raw_transcript'].fillna("").tolist()
    emb_text = st_model.encode(raw_texts, show_progress_bar=False, batch_size=32)
    np.save(cache_path, emb_text)
    return emb_text
