import os
import pandas as pd
import numpy as np
import torch
import laion_clap
import glob
from tqdm import tqdm
from scipy import linalg
import mauve
import soundfile as sf

# ==========================================
# CONFIGURATION
# ==========================================
# USER MUST SET THESE PATHS ON THE SERVER
PATH_TO_MTG_JAMENDO = "../datasets/mtg-jamendo" 
PATH_TO_FMA_POP = "../datasets/fma-pop"
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
    with torch.no_grad():
        for i in tqdm(range(0, len(audio_files), batch_size), leave=False):
            batch_files = audio_files[i:i+batch_size]
            try:
                # get_audio_embedding_from_filelist handles loading and resampling
                # Returns (B, D)
                batch_embed = model.get_audio_embedding_from_filelist(x=batch_files, use_tensor=True)
                embeddings.append(batch_embed.detach().cpu().numpy())
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
            # waveform, sr = torchaudio.load(file_path)
            # Use soundfile directly to avoid backend issues
            data, sr = sf.read(file_path)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0) # (1, Time)
            else:
                waveform = waveform.t() # (Channels, Time)
                # Mix to mono
                waveform = torch.mean(waveform, dim=0, keepdim=True) # (1, Time)
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
    
    # 1. Load Metadata and Responses
    try:
        df_meta = pd.read_csv("AImoclips/clips_metadata.csv")
        df_responses = pd.read_csv("analysis/combined_responses.csv")
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}")
        return

    # Pre-calculate clip-wise mean valence and arousal
    print("Calculating clip-wise mean ratings...")
    clip_mean_ratings = df_responses.groupby(['model', 'audio_file']).agg({
        'valence': 'mean',
        'arousal': 'mean'
    }).reset_index()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    bootstrapped_output_csv = "analysis/metrics/bootstrapped_distribution_metrics.csv"
    
    models = df_meta['model'].unique()
    
    # 2. Initialize CLAP (for FAD)
    clap_model = None
    mu_ref, sigma_ref = None, None
    
    print("\nInitializing CLAP model (for FAD)...")
    try:
        clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        ckpt_path = "music_audioset_epoch_15_esc_90.14.pt"
        if os.path.exists(ckpt_path):
            clap_model.load_ckpt(ckpt=ckpt_path)
        else:
            print("Warning: Local checkpoint not found. Using default.")
            clap_model.load_ckpt()
            
        # FAD Reference: MTG-Jamendo (CLAP Embeddings)
        print(f"\nProcessing FAD Reference (MTG-Jamendo) from {PATH_TO_MTG_JAMENDO}...")
        ref_files_fad = get_audio_files(PATH_TO_MTG_JAMENDO, max_files=2000)
        if ref_files_fad:
            ref_embed_fad = get_clap_embeddings(ref_files_fad, clap_model)
            if ref_embed_fad is not None:
                mu_ref, sigma_ref = np.mean(ref_embed_fad, axis=0), np.cov(ref_embed_fad, rowvar=False)
    except Exception as e:
        print(f"Error initializing CLAP: {e}")
        return

    results = []
    
    # 4. Process Each Model with Bootstrapping
    N_BOOTSTRAP = 30
    SAMPLE_SIZE = 50

    for model_name in tqdm(models, desc="Processing Models"):
        print(f"\nProcessing Model: {model_name}...")
        
        # Get all files for this model
        model_dir = os.path.join(AUDIO_CLIPS_DIR, model_name)
        if not os.path.exists(model_dir):
            print(f"Warning: Directory for {model_name} not found.")
            continue
            
        all_model_files = get_audio_files(model_dir)
        if len(all_model_files) < 10:
             print(f"Warning: Not enough files for {model_name}. Skipping calculation.")
             continue

        # Pre-calculate embeddings for ALL files to save time
        print(f"  Pre-calculating embeddings for {len(all_model_files)} files...")
        all_embeddings = get_clap_embeddings(all_model_files, clap_model)
        
        # Create a mapping from filename to embedding index
        # We need to map basename because 'audio_file' in csv is basename
        file_basename_to_idx = {os.path.basename(f): i for i, f in enumerate(all_model_files)}
        
        # Filter clip_mean_ratings for this model
        model_ratings = clip_mean_ratings[clip_mean_ratings['model'] == model_name].copy()
        
        # FIX: The CSV filenames might have '_crop' suffix (e.g., 'file_crop.wav') 
        # but disk files might not. We try to normalize to match.
        def normalize_name(fname):
            return fname.replace('_crop', '')
            
        model_ratings['normalized_file'] = model_ratings['audio_file'].apply(normalize_name)
        
        # Only keep files that exist in both embeddings and ratings (using normalized name for checking)
        # We need to map the normalized name back to the original CSV name to retrieve ratings later
        # Create a lookup: normalized_name -> original_csv_name
        valid_files_map = {} 
        for csv_file, norm_file in zip(model_ratings['audio_file'], model_ratings['normalized_file']):
             if norm_file in file_basename_to_idx:
                 valid_files_map[norm_file] = csv_file
                 
        valid_files_disk = list(valid_files_map.keys())
        
        if len(valid_files_disk) < SAMPLE_SIZE:
             print(f"Warning: Not enough valid files with ratings for {model_name}. Found {len(valid_files_disk)}.")
             # We can still proceed if we sample with replacement from what we have, 
             # provided we have at least some files.
             if len(valid_files_disk) == 0:
                 continue
        
        print(f"  Bootstrapping over {len(valid_files_disk)} valid files...")

        for i in tqdm(range(N_BOOTSTRAP)):
             # Sample files with replacement (using disk filenames)
            sampled_disk_files = np.random.choice(valid_files_disk, SAMPLE_SIZE, replace=True)
            
            # 1. Calculate Mean Valence & Arousal
            # We need to look up the original CSV filename to get the rating
            
            # Efficient way: map original_csv_filename to (val, aro)
            file_to_rating = model_ratings.set_index('audio_file')[['valence', 'arousal']].to_dict('index')
            
            sampled_valence = []
            sampled_arousal = []
            
            for f_disk in sampled_disk_files:
                original_csv_name = valid_files_map[f_disk]
                sampled_valence.append(file_to_rating[original_csv_name]['valence'])
                sampled_arousal.append(file_to_rating[original_csv_name]['arousal'])
            
            mean_valence = np.mean(sampled_valence)
            mean_arousal = np.mean(sampled_arousal)
            
            # 2. Calculate FAD
            # Get integer indices for embeddings
            embed_indices = [file_basename_to_idx[f] for f in sampled_disk_files]
            sampled_embeddings = all_embeddings[embed_indices]
            
            mu_model, sigma_model = np.mean(sampled_embeddings, axis=0), np.cov(sampled_embeddings, rowvar=False)
            
            fad_score = calculate_frechet_distance(mu_model, sigma_model, mu_ref, sigma_ref)
            
            results.append({
                'model': model_name,
                'iteration': i,
                'fad_score': fad_score,
                'valence': mean_valence,
                'arousal': mean_arousal
            })

    # Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv(bootstrapped_output_csv, index=False)
    print(f"\nSaved bootstrapped metrics to {bootstrapped_output_csv}")

if __name__ == "__main__":
    main()
