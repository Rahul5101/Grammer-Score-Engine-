import os
import re
import math
import numpy as np
import pandas as pd
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from .config import CONFIG

def extract_text_features(transcripts_df):
    """Extracts linguistic, syntactic, acceptability (CoLA), perplexity (GPT-2), and disfluency features."""
    cache_path = os.path.join(CONFIG['cache_dir'], "feats_text.parquet")
    
    if os.path.exists(cache_path):
        feats_df = pd.read_parquet(cache_path)
        if 'split' in feats_df.columns:
            feats_df = feats_df.drop_duplicates(subset=['split', 'filename']).reset_index(drop=True)
            cached_keys = set(zip(feats_df['split'], feats_df['filename']))
            req_keys = set(zip(transcripts_df['split'], transcripts_df['filename']))
            if len(feats_df) == len(transcripts_df) and cached_keys == req_keys:
                print("Loading cached text features...")
                return feats_df
        else:
            print("Cached text features missing 'split' column. Recomputing...")
        print(f"Cached text features mismatch or incomplete ({len(feats_df)} vs {len(transcripts_df)}). Recomputing...")

    print("Extracting Text & Linguistic Features (spaCy, CoLA, GPT-2)...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading missing spaCy model 'en_core_web_sm'...", flush=True)
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    cola_name = "textattack/roberta-base-CoLA"
    cola_tok = AutoTokenizer.from_pretrained(cola_name)
    cola_mod = AutoModelForSequenceClassification.from_pretrained(cola_name).eval()

    gpt2_name = "gpt2"
    gpt2_tok = GPT2Tokenizer.from_pretrained(gpt2_name)
    gpt2_mod = GPT2LMHeadModel.from_pretrained(gpt2_name).eval()

    fillers_set = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'mean'}
    text_feat_list = []

    for idx, row in transcripts_df.iterrows():
        fname = row['filename']
        split_name = row['split']
        raw_text = row['raw_transcript']
        clean_text = row['clean_transcript']
        words = clean_text.split()
        num_words = max(len(words), 1)

        # Grammar error heuristics
        sv_errors = len(re.findall(r'\b(he|she|it)\s+(go|have|do|were|are)\b', clean_text))
        sv_errors += len(re.findall(r'\b(they|we|you)\s+(is|was|has|goes)\b', clean_text))
        double_negs = len(re.findall(r'\b(don\'t|doesn\'t|didn\'t|never|cannot)\s+\w+\s+(no|nothing|nobody|none)\b', clean_text))
        rep_adjacent = len(re.findall(r'\b(\w+)\s+\1\b', clean_text))
        lt_total = sv_errors + double_negs + rep_adjacent
        lt_rate = (lt_total / num_words) * 100

        # spaCy syntax features
        doc = nlp(raw_text if len(raw_text) > 0 else "N/A")
        sents = list(doc.sents)
        sent_count = len(sents) if len(sents) > 0 else 1
        sent_lens = [len([t for t in s if not t.is_punct]) for s in sents]
        mean_sent_len = float(np.mean(sent_lens))
        max_sent_len = float(np.max(sent_lens))
        std_sent_len = float(np.std(sent_lens))

        complete_sents = 0
        dep_depths = []
        subclause_count = 0
        passive_count = 0

        for s in sents:
            has_verb = any(t.pos_ in ('VERB', 'AUX') for t in s)
            has_subj = any('subj' in t.dep_ for t in s)
            if has_verb and has_subj:
                complete_sents += 1
            for t in s:
                depth = 1
                curr = t
                while curr.head != curr:
                    depth += 1
                    curr = curr.head
                dep_depths.append(depth)
                if t.dep_ in ('advcl', 'relcl', 'acl'):
                    subclause_count += 1
                if t.dep_ in ('auxpass', 'nsubjpass'):
                    passive_count += 1

        completeness_ratio = complete_sents / sent_count
        mean_dep_depth = float(np.mean(dep_depths)) if len(dep_depths) > 0 else 1.0
        max_dep_depth = float(np.max(dep_depths)) if len(dep_depths) > 0 else 1.0

        pos_counts = doc.count_by(spacy.attrs.POS)
        total_toks = max(len(doc), 1)
        pos_noun_ratio = (pos_counts.get(spacy.symbols.NOUN, 0) + pos_counts.get(spacy.symbols.PROPN, 0)) / total_toks
        pos_verb_ratio = (pos_counts.get(spacy.symbols.VERB, 0) + pos_counts.get(spacy.symbols.AUX, 0)) / total_toks
        pos_adj_ratio = pos_counts.get(spacy.symbols.ADJ, 0) / total_toks
        pos_adv_ratio = pos_counts.get(spacy.symbols.ADV, 0) / total_toks
        pos_pron_ratio = pos_counts.get(spacy.symbols.PRON, 0) / total_toks

        unique_tokens = len(set([t.text.lower() for t in doc if t.is_alpha]))
        ttr = unique_tokens / max(len([t for t in doc if t.is_alpha]), 1)

        # CoLA Acceptability
        cola_probs = []
        for s in sents[:5]:
            stext = s.text.strip()
            if len(stext) > 3:
                in_c = cola_tok(stext, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    logits = cola_mod(**in_c).logits
                    probs = torch.softmax(logits, dim=-1)
                    cola_probs.append(probs[0, 1].item())
        if len(cola_probs) == 0:
            cola_probs = [0.5]
        mean_cola = float(np.mean(cola_probs))
        min_cola = float(np.min(cola_probs))
        std_cola = float(np.std(cola_probs))
        frac_cola_below_05 = float(np.mean([p < 0.5 for p in cola_probs]))

        # GPT-2 Perplexity
        gpt2_losses = []
        for s in sents[:3]:
            stext = s.text.strip()
            if len(stext) > 5:
                enc = gpt2_tok(stext, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    loss = gpt2_mod(**enc, labels=enc["input_ids"]).loss
                    gpt2_losses.append(loss.item())
        if len(gpt2_losses) == 0:
            gpt2_losses = [3.5]
        mean_gpt2_loss = float(np.mean(gpt2_losses))
        max_gpt2_loss = float(np.max(gpt2_losses))
        gpt2_ppl = math.exp(min(mean_gpt2_loss, 10.0))

        filler_count = sum(1 for w in words if w in fillers_set)
        filler_rate = (filler_count / num_words) * 100
        repetition_rate = (rep_adjacent / num_words) * 100

        text_feat_list.append({
            'filename': fname,
            'split': split_name,
            'lt_total_errors': lt_total,
            'lt_errors_per_100': lt_rate,
            'spacy_sent_count': sent_count,
            'spacy_mean_sent_len': mean_sent_len,
            'spacy_max_sent_len': max_sent_len,
            'spacy_std_sent_len': std_sent_len,
            'spacy_completeness_ratio': completeness_ratio,
            'spacy_mean_dep_depth': mean_dep_depth,
            'spacy_max_dep_depth': max_dep_depth,
            'spacy_subclause_count': subclause_count,
            'spacy_passive_count': passive_count,
            'spacy_pos_noun_ratio': pos_noun_ratio,
            'spacy_pos_verb_ratio': pos_verb_ratio,
            'spacy_pos_adj_ratio': pos_adj_ratio,
            'spacy_pos_adv_ratio': pos_adv_ratio,
            'spacy_pos_pron_ratio': pos_pron_ratio,
            'spacy_ttr': ttr,
            'cola_mean_prob': mean_cola,
            'cola_min_prob': min_cola,
            'cola_std_prob': std_cola,
            'cola_frac_below_05': frac_cola_below_05,
            'gpt2_mean_loss': mean_gpt2_loss,
            'gpt2_max_loss': max_gpt2_loss,
            'gpt2_ppl': gpt2_ppl,
            'disfluency_filler_count': filler_count,
            'disfluency_filler_rate': filler_rate,
            'disfluency_repetition_rate': repetition_rate
        })

    feats_text_df = pd.DataFrame(text_feat_list).drop_duplicates(subset=['split', 'filename']).reset_index(drop=True)
    feats_text_df.to_parquet(cache_path)
    return feats_text_df
