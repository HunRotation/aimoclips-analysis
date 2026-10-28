import os
import requests
import zipfile
import subprocess
import pandas as pd
import shutil
from tqdm import tqdm

# Configuration
DATASETS_DIR = "../datasets"
CLAP_CKPT_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
CLAP_CKPT_PATH = "AImoclips/music_audioset_epoch_15_esc_90.14.pt"
FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
MTG_REPO_URL = "https://github.com/MTG/mtg-jamendo-dataset.git"

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file.")

def setup_fma_pop():
    fma_pop_dir = os.path.join(DATASETS_DIR, "fma-pop")
    if os.path.exists(fma_pop_dir) and len(os.listdir(fma_pop_dir)) > 10:
        print(f"FMA-Pop directory seems populated: {fma_pop_dir}")
        return

    # Download FMA Small and Metadata
    os.makedirs(DATASETS_DIR, exist_ok=True)
    fma_small_zip = os.path.join(DATASETS_DIR, "fma_small.zip")
    fma_meta_zip = os.path.join(DATASETS_DIR, "fma_metadata.zip")
    
    download_file(FMA_SMALL_URL, fma_small_zip)
    download_file(FMA_META_URL, fma_meta_zip)
    
    # Unzip
    if os.path.exists(fma_small_zip):
        unzip_file(fma_small_zip, DATASETS_DIR)
    if os.path.exists(fma_meta_zip):
        unzip_file(fma_meta_zip, DATASETS_DIR)
    
    # Filter for Pop
    print("Filtering for Pop tracks...")
    tracks_csv = os.path.join(DATASETS_DIR, "fma_metadata", "tracks.csv")
    if not os.path.exists(tracks_csv):
        print(f"Error: {tracks_csv} not found.")
        return

    # tracks.csv has 3 header rows. Row 0 is 'track', 'album', etc. Row 1 is 'genre_top', etc.
    # We want ('track', 'genre_top')
    try:
        tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
        
        # Filter: subset='small' AND genre_top='Pop'
        # The 'subset' column is ('set', 'subset')
        small_subset = tracks[tracks[('set', 'subset')] == 'small']
        pop_tracks = small_subset[small_subset[('track', 'genre_top')] == 'Pop']
        
        print(f"Found {len(pop_tracks)} Pop tracks in FMA Small.")
        
        # Copy files
        os.makedirs(fma_pop_dir, exist_ok=True)
        fma_small_dir = os.path.join(DATASETS_DIR, "fma_small")
        
        count = 0
        for track_id in tqdm(pop_tracks.index, desc="Copying Pop tracks"):
            # Track ID is integer, file path is fma_small/000/000123.mp3
            tid_str = f"{track_id:06d}"
            folder = tid_str[:3]
            src_path = os.path.join(fma_small_dir, folder, f"{tid_str}.mp3")
            
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(fma_pop_dir, f"{tid_str}.mp3"))
                count += 1
            else:
                # Try .wav just in case, though FMA small is mp3
                pass
                
        print(f"Copied {count} files to {fma_pop_dir}")
    except Exception as e:
        print(f"Error processing FMA metadata: {e}")

def setup_mtg_jamendo():
    mtg_dir = os.path.join(DATASETS_DIR, "mtg-jamendo")
    if os.path.exists(mtg_dir) and len(os.listdir(mtg_dir)) > 10:
        print(f"MTG-Jamendo directory seems populated: {mtg_dir}")
        return

    print("Setting up MTG-Jamendo...")
    os.makedirs(mtg_dir, exist_ok=True)
    
    # Let's download the raw_30s metadata TSV
    # URL found via search: https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw_30s_cleantags.tsv
    tsv_url = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw_30s_cleantags.tsv"
    tsv_path = os.path.join(DATASETS_DIR, "raw_30s_cleantags.tsv")
    download_file(tsv_url, tsv_path)
    
    if not os.path.exists(tsv_path):
        print("Failed to download MTG-Jamendo metadata.")
        return

    try:
        df = pd.read_csv(tsv_path, sep='\t')
        # Columns: track_id, artist_id, album_id, path, duration, tags
        # Path example: 00/1009600.mp3
        # Base URL for raw_30s: https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/
        
        base_url = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/"
        
        # Sample 2000 tracks
        sample = df.sample(n=2000, random_state=42)
        
        print(f"Downloading 2000 tracks from MTG-Jamendo...")
        for _, row in tqdm(sample.iterrows(), total=len(sample)):
            rel_path = row['path'] # e.g. 00/12345.mp3
            file_url = base_url + rel_path
            dest_path = os.path.join(mtg_dir, os.path.basename(rel_path))
            
            if not os.path.exists(dest_path):
                try:
                    # Use a simple get with timeout
                    r = requests.get(file_url, stream=True, timeout=10)
                    if r.status_code == 200:
                        with open(dest_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=1024):
                                f.write(chunk)
                except Exception as e:
                    print(f"Failed to download {rel_path}: {e}")
    except Exception as e:
        print(f"Error processing MTG-Jamendo: {e}")

def main():
    # 1. CLAP Checkpoint
    download_file(CLAP_CKPT_URL, CLAP_CKPT_PATH)
    
    # 2. FMA-Pop
    # setup_fma_pop() # Skipped due to slow download speed
    
    # 3. MTG-Jamendo
    setup_mtg_jamendo()
    
    print("\nSetup complete!")

if __name__ == "__main__":
    main()
