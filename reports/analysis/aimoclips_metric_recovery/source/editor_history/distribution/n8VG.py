import os
import pandas as pd
import numpy as np
import torch
import laion_clap
import glob
from tqdm import tqdm
from scipy import linalg
import mauve

# ==========================================
# CONFIGURATION
# ==========================================
# USER MUST SET THESE PATHS ON THE SERVER
PATH_TO_MTG_JAMENDO = "/path/to/mtg-jamendo-dataset" 
PATH_TO_FMA_POP = "/path/to/fma-pop-dataset"
AUDIO_CLIPS_DIR = "AImoclips/music_clips"
OUTPUT_CSV = "analysis/metrics/distribution_metrics.csv"

if not os.path.exists(PATH_TO_MTG_JAMENDO):
    os.makedirs(PATH_TO_MTG_JAMENDO)
if not os.path.exists(PATH_TO_FMA_POP):
    os.makedirs(PATH_TO_FMA_POP)
if not os.path.exists(AUDIO_CLIPS_DIR):
    os.makedirs(AUDIO_CLIPS_DIR)
if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
    os.makedirs(os.path.dirname(OUTPUT_CSV))

# ==========================================
# UTILS
# ==========================================
def get_audio_files(directory, max_files=None):
    """Recursively find audio files."""
    files = glob.glob(os.path.join(directory, "**/*.mp3"), recursive=True) + \
            glob.glob(os.path.join(directory, "**/*.wav"), recursive=True)
    if max_files and len(files) > max_files:
        files = np.random.choice(files, max_files, replace=False).tolist()
    return files

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

# ==========================================
# EMBEDDING EXTRACTION
# ==========================================
def get_clap_embeddings(audio_files, model):
    """Extract CLAP embeddings (for FAD)."""
    embeddings = []
    batch_size = 16
    
    print(f"Extracting CLAP embeddings for {len(audio_files)} files...")
    for i in tqdm(range(0, len(audio_files), batch_size), leave=False):
        batch_files = audio_files[i:i+batch_size]
        try:
            # get_audio_embedding_from_file_list handles loading and resampling
            # Returns (B, D)
            batch_embed = model.get_audio_embedding_from_file_list(x=batch_files, use_tensor=True)
            embeddings.append(batch_embed.cpu().numpy())
        except Exception as e:
            print(f"Error processing CLAP batch {i}: {e}")
            
    if not embeddings:
        return None
    return np.concatenate(embeddings, axis=0)

def get_mert_embeddings(audio_files):
    """Extract MERT embeddings (for MAD). Uses MERT-v1-330M."""
    try:
        from transformers import Wav2Vec2FeatureExtractor, AutoModel
        import torchaudio
    except ImportError:
        print("Error: transformers or torchaudio not found.")
        return None

    model_id = "m-a-p/MERT-v1-330M"
    print(f"Loading MERT model: {model_id}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    except Exception as e:
        print(f"Error loading MERT: {e}")
        return None

    embeddings = []
    sampling_rate = processor.sampling_rate
    
    print(f"Extracting MERT embeddings for {len(audio_files)} files...")
    for file_path in tqdm(audio_files, leave=False):
        try:
            waveform, sr = torchaudio.load(file_path)
            if sr != sampling_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
            
            # MERT expects input roughly 5s-10s? 
            # We'll take up to 10s to be safe/consistent
            if waveform.shape[1] > sampling_rate * 10:
                waveform = waveform[:, :sampling_rate * 10]
                
            inputs = processor(waveform.squeeze(), sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Use the last hidden state, averaged over time (or specific layer as per paper?)
            # Standard practice for MERT is often averaging last layer
            all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
            # shape: (layers, time, dim). We usually take the last layer.
            last_hidden = outputs.last_hidden_state # (B, Time, Dim)
            
            # Mean pooling
            embed = last_hidden.mean(dim=1).cpu().numpy()
            embeddings.append(embed)
            
        except Exception as e:
            print(f"Error processing MERT file {file_path}: {e}")

    if not embeddings:
        return None
    return np.concatenate(embeddings, axis=0)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Starting Distribution Metrics Calculation (FAD & MAD)...")
    
    # 1. Load Metadata
    try:
        df_meta = pd.read_csv("AImoclips/clips_metadata.csv")
    except FileNotFoundError:
        print("Error: clips_metadata.csv not found.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # 2. Initialize CLAP (for FAD)
    print("\nInitializing CLAP model (for FAD)...")
    try:
        clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        ckpt_path = "music_audioset_epoch_15_esc_90.14.pt"
        if os.path.exists(ckpt_path):
            clap_model.load_ckpt(ckpt=ckpt_path)
        else:
            print("Warning: Local checkpoint not found. Using default.")
            clap_model.load_ckpt()
    except Exception as e:
        print(f"Error initializing CLAP: {e}")
        return

    # 3. Process Reference Datasets
    # FAD Reference: MTG-Jamendo (CLAP Embeddings)
    print(f"\nProcessing FAD Reference (MTG-Jamendo) from {PATH_TO_MTG_JAMENDO}...")
    ref_files_fad = get_audio_files(PATH_TO_MTG_JAMENDO, max_files=2000) # Limit for speed if needed
    if ref_files_fad:
        ref_embed_fad = get_clap_embeddings(ref_files_fad, clap_model)
        if ref_embed_fad is not None:
            mu_ref, sigma_ref = np.mean(ref_embed_fad, axis=0), np.cov(ref_embed_fad, rowvar=False)
        else:
            mu_ref, sigma_ref = None, None
    else:
        print("Warning: No FAD reference files found.")
        mu_ref, sigma_ref = None, None

    # MAD Reference: FMA-Pop (MERT Embeddings)
    print(f"\nProcessing MAD Reference (FMA-Pop) from {PATH_TO_FMA_POP}...")
    ref_files_mad = get_audio_files(PATH_TO_FMA_POP, max_files=2000)
    if ref_files_mad:
        ref_embed_mad = get_mert_embeddings(ref_files_mad) # This loads MERT inside
    else:
        print("Warning: No MAD reference files found.")
        ref_embed_mad = None

    results = []
    
    # 4. Process Each Model
    models = df_meta['model'].unique()
    
    for model_name in tqdm(models, desc="Processing Models"):
        print(f"\nProcessing Model: {model_name}...")
        
        # Get files for this model
        model_files = []
        # Assuming folder structure AImoclips/music_clips/{Model}
        model_dir = os.path.join(AUDIO_CLIPS_DIR, model_name)
        if os.path.exists(model_dir):
            model_files = get_audio_files(model_dir)
        
        if len(model_files) < 10:
            print(f"Warning: Not enough files for {model_name}. Skipping.")
            continue
            
        # --- Calculate FAD (CLAP) ---
        fad_score = None
        if mu_ref is not None:
            print("  Calculating FAD (CLAP)...")
            model_embed_fad = get_clap_embeddings(model_files, clap_model)
            if model_embed_fad is not None:
                mu_model, sigma_model = np.mean(model_embed_fad, axis=0), np.cov(model_embed_fad, rowvar=False)
                fad_score = calculate_frechet_distance(mu_model, sigma_model, mu_ref, sigma_ref)
                print(f"  FAD: {fad_score:.4f}")
        
        # --- Calculate MAD (MERT) ---
        mad_score = None
        if ref_embed_mad is not None:
            print("  Calculating MAD (MERT)...")
            # Re-load MERT or keep it loaded? 
            # Ideally load once, but get_mert_embeddings loads it. 
            # Let's optimize if needed, but for now calling function is safer for memory if we clear it.
            # Actually, better to keep it loaded if possible, but the function handles it.
            model_embed_mad = get_mert_embeddings(model_files)
            if model_embed_mad is not None:
                out = mauve.compute_mauve(p_features=ref_embed_mad, q_features=model_embed_mad, max_text_length=256, verbose=False)
                mad_score = out.mauve
                print(f"  MAD: {mad_score:.4f}")

        results.append({
            'model': model_name,
            'fad_score': fad_score,
            'mad_score': mad_score
        })

    # Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved distribution metrics to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
