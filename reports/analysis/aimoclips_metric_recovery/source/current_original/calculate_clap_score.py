import pandas as pd
import numpy as np
import os
import torch
import laion_clap
from tqdm import tqdm

def calculate_clap_scores():
    print("Starting CLAP Score Calculation...")

    # 1. Load Data
    csv_path = 'analysis/combined_responses.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} responses.")

    # 2. Get Unique Clips
    # We only need to calculate score once per clip
    unique_clips = df[['audio_file', 'model', 'emotion']].drop_duplicates()
    print(f"Found {len(unique_clips)} unique clips.")

    # 3. Load CLAP Model
    # User requested LAION-MA. usually 'music_audioset_epoch_15_esc_90.14.pt' or similar
    # laion_clap provides a wrapper.
    print("Loading CLAP model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # quantization=False for better accuracy if memory allows
    # User requested music_audioset checkpoint
    # Trying enable_fusion=False as some music checkpoints might be audio-only or have different structure.
    # Also reverting to HTSAT-base.
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    
    ckpt_path = "music_audioset_epoch_15_esc_90.14.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading local checkpoint: {ckpt_path}")
        model.load_ckpt(ckpt=ckpt_path)
    else:
        print("Checkpoint not found, downloading default...")
        model.load_ckpt()
    # Note: For music-audio specifically, we might want a specific checkpoint, 
    # but the default large one is usually good. 
    # Let's try to load 'music_audioset_epoch_15_esc_90.14.pt' if possible, 
    # or just use the default which is robust. 
    # The library defaults to 630k-fusion-best.
    
    model.to(device)
    model.eval()

    results = []
    
    # Base path for audio
    audio_base_path = 'AImoclips/music_clips'

    print("Calculating scores...")
    for index, row in tqdm(unique_clips.iterrows()):
        audio_file = row['audio_file']
        model_name = row['model']
        emotion = row['emotion']
        
        # Construct path
        # Structure: AImoclips/music_clips/{Model}/{Filename}
        # Note: 'model' in CSV matches directory name? 
        # CSV: Mustango, Udio, Suno, AudioLDM2, StableAudio, MusicGen
        # Dir: Mustango, Udio, Suno, AudioLDM2, StableAudio, MusicGen
        # Checked directory listing earlier, they match.
        
        # FIX: CSV has '_crop.wav' but files on disk are '.wav'
        actual_filename = audio_file.replace('_crop.wav', '.wav')
        
        audio_path = os.path.join(audio_base_path, model_name, actual_filename)
        
        if not os.path.exists(audio_path):
            print(f"Warning: File not found {audio_path}")
            continue
            
        # Text Prompt
        text_prompt = [f"{emotion}"]
        
        try:
            # Get Embeddings
            # audio_file can be a list or single path
            with torch.no_grad():
                # audio_embed: [1, 512], text_embed: [1, 512]
                audio_embed = model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
                text_embed = model.get_text_embedding(text_prompt, use_tensor=True)
                
                # Cosine Similarity
                # Normalize first
                audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                
                score = (audio_embed @ text_embed.T).item()
                
                # Scale to 0-1? Cosine sim is -1 to 1. 
                # Usually we keep it as is or clamp to 0-1. 
                # Let's keep raw cosine similarity.
                
                results.append({
                    'audio_file': audio_file,
                    'clap_score': score
                })
                
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    # 4. Save Results
    output_dir = 'analysis/metrics'
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'clap_scores.csv'), index=False)
    print(f"Saved {len(results_df)} scores to {output_dir}/clap_scores.csv")

if __name__ == "__main__":
    calculate_clap_scores()
