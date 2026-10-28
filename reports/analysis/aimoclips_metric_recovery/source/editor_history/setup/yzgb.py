import os
import requests
import zipfile
import pandas as pd
import shutil
import subprocess
import glob
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "datasets")
CLAP_CKPT_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
CLAP_CKPT_PATH = os.path.join(PROJECT_ROOT, "music_audioset_epoch_15_esc_90.14.pt")

FMA_POP_CSV_URL = "https://raw.githubusercontent.com/microsoft/fadtk/master/fadtk/fma_pop_tracks.csv"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_MEDIUM_URL = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"

MTG_REPO_URL = "https://github.com/MTG/mtg-jamendo-dataset.git"

# ==========================================
# UTILS
# ==========================================
def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# ==========================================
# TASKS
# ==========================================
def setup_directories():
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)
        print(f"Created datasets directory: {DATASETS_DIR}")

def download_clap():
    print("\n--- Downloading CLAP Checkpoint ---")
    download_file(CLAP_CKPT_URL, CLAP_CKPT_PATH)

def prepare_fma_pop():
    print("\n--- Preparing FMA-Pop Dataset ---")
    fma_pop_dir = os.path.join(DATASETS_DIR, "fma-pop")
    if os.path.exists(fma_pop_dir) and len(os.listdir(fma_pop_dir)) > 100:
        print("FMA-Pop directory already exists and seems populated. Skipping.")
        return

    # 1. Download fma_pop_tracks.csv
    fma_pop_csv_path = os.path.join(DATASETS_DIR, "fma_pop_tracks.csv")
    download_file(FMA_POP_CSV_URL, fma_pop_csv_path)

    # 2. Download and Extract Metadata
    fma_metadata_zip = os.path.join(DATASETS_DIR, "fma_metadata.zip")
    download_file(FMA_METADATA_URL, fma_metadata_zip)
    if not os.path.exists(os.path.join(DATASETS_DIR, "fma_metadata")):
        extract_zip(fma_metadata_zip, DATASETS_DIR)

    # 3. Load Metadata and Filter
    print("Loading metadata...")
    tracks = pd.read_csv(os.path.join(DATASETS_DIR, "fma_metadata", "tracks.csv"), index_col=0, header=[0, 1])
    fma_pop_tracks = pd.read_csv(fma_pop_csv_path)
    
    # Filter based on excluded genres
    # Excluded: "Spoken", "Poetry", "Soundtrack", "Composed Music"
    # We need to check the 'track' -> 'genres' or 'genre_top' in the main tracks.csv
    # But fma_pop_tracks.csv might just be IDs. Let's check.
    # Assuming fma_pop_tracks.csv has 'track_id'
    
    pop_track_ids = fma_pop_tracks['track_id'].values
    
    # Filter out excluded genres
    # We need to look up genres for these IDs.
    # The 'tracks.csv' has genre information.
    
    # Let's verify which subset we need.
    # Check if all pop_track_ids are in fma_small
    subset_small = tracks[tracks[('set', 'subset')] == 'small']
    subset_medium = tracks[tracks[('set', 'subset')].isin(['small', 'medium'])]
    
    missing_in_small = [tid for tid in pop_track_ids if tid not in subset_small.index]
    
    if len(missing_in_small) == 0:
        print("All FMA-Pop tracks are in FMA-Small.")
        target_zip_url = FMA_SMALL_URL
        target_zip_name = "fma_small.zip"
        extract_dir_name = "fma_small"
    else:
        print(f"Found {len(missing_in_small)} tracks missing from FMA-Small. Checking Medium...")
        missing_in_medium = [tid for tid in pop_track_ids if tid not in subset_medium.index]
        if len(missing_in_medium) == 0:
             print("All FMA-Pop tracks are in FMA-Medium.")
             target_zip_url = FMA_MEDIUM_URL
             target_zip_name = "fma_medium.zip"
             extract_dir_name = "fma_medium"
        else:
            print(f"Warning: {len(missing_in_medium)} tracks missing even from Medium. Defaulting to Medium and hoping for the best.")
            target_zip_url = FMA_MEDIUM_URL
            target_zip_name = "fma_medium.zip"
            extract_dir_name = "fma_medium"

    # 4. Download Audio
    fma_zip_path = os.path.join(DATASETS_DIR, target_zip_name)
    download_file(target_zip_url, fma_zip_path)
    
    if not os.path.exists(os.path.join(DATASETS_DIR, extract_dir_name)):
        extract_zip(fma_zip_path, DATASETS_DIR)

    # 5. Copy and Filter Files
    if not os.path.exists(fma_pop_dir):
        os.makedirs(fma_pop_dir)

    print("Filtering and copying files...")
    # Load genres to filter
    genres = pd.read_csv(os.path.join(DATASETS_DIR, "fma_metadata", "genres.csv"), index_col=0)
    
    # We need to identify IDs that belong to excluded genres.
    # The paper says excluded: "Spoken", "Poetry", "Soundtrack", "Composed Music"
    # We can check the 'track' -> 'genres_all' column in tracks.csv which contains lists of genre IDs.
    # Or we can check 'track' -> 'genre_top'.
    # Let's try to be precise.
    
    # Get genre IDs for excluded genres
    excluded_genre_names = ["Spoken", "Poetry", "Soundtrack", "Composed Music"]
    excluded_genre_ids = genres[genres['title'].isin(excluded_genre_names)].index.tolist()
    
    count_copied = 0
    for tid in tqdm(pop_track_ids):
        if tid not in tracks.index:
            continue
            
        track_genres = tracks.loc[tid, ('track', 'genres_all')]
        # genres_all is a string representation of a list, e.g. "[1, 2, 3]"
        # We need to parse it safely
        try:
            current_genre_ids = eval(track_genres)
        except:
            current_genre_ids = []
            
        # Check if any excluded genre is present
        if any(gid in excluded_genre_ids for gid in current_genre_ids):
            continue
            
        # Find the file
        # FMA structure: fma_small/000/000123.mp3
        tid_str = f"{tid:06d}"
        src_path = os.path.join(DATASETS_DIR, extract_dir_name, tid_str[:3], f"{tid_str}.mp3")
        
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(fma_pop_dir, f"{tid_str}.mp3"))
            count_copied += 1
            
    print(f"Copied {count_copied} tracks to {fma_pop_dir}")


def download_mtg_jamendo():
    print("\n--- Downloading MTG-Jamendo Subset ---")
    mtg_dir = os.path.join(DATASETS_DIR, "mtg-jamendo")
    if os.path.exists(mtg_dir) and len(os.listdir(mtg_dir)) > 100:
        print("MTG-Jamendo directory already exists and seems populated. Skipping.")
        return

    repo_dir = os.path.join(DATASETS_DIR, "mtg-jamendo-dataset")
    if not os.path.exists(repo_dir):
        print("Cloning MTG-Jamendo repository...")
        subprocess.run(["git", "clone", MTG_REPO_URL, repo_dir], check=True)

    # We need to run the download script.
    # The repo has scripts/download/download.py usually.
    # Let's assume we can just use a simple custom downloader for a subset to avoid complex dependencies.
    # The dataset is hosted on common URLs.
    # We need a list of files. 
    # The repo contains metadata/tracks.tsv
    
    print("Reading MTG metadata...")
    # We might need to download the metadata file first if it's not in the clone (it usually is)
    # Actually, let's just use the repo's instructions if possible, but for a subset, custom is better.
    
    # For now, let's try to find the metadata file in the cloned repo
    # If not found, we will download it.
    
    # URL for raw_30s metadata
    metadata_url = "https://github.com/MTG/mtg-jamendo-dataset/raw/master/data/autotagging_moodtheme.tsv"
    metadata_path = os.path.join(DATASETS_DIR, "mtg_metadata.tsv")
    download_file(metadata_url, metadata_path)
    
    df_mtg = pd.read_csv(metadata_path, sep='\t')
    
    # Select 2000 random tracks
    subset = df_mtg.sample(n=2000, random_state=42)
    
    if not os.path.exists(mtg_dir):
        os.makedirs(mtg_dir)
        
    print(f"Downloading 2000 tracks to {mtg_dir}...")
    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        path = row['PATH'] # e.g. 00/12345.mp3
        url = f"https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/{path}"
        
        dest_path = os.path.join(mtg_dir, os.path.basename(path))
        try:
            # We can use a simple check to see if it exists
            if not os.path.exists(dest_path):
                # Use requests to download
                r = requests.get(url)
                if r.status_code == 200:
                    with open(dest_path, 'wb') as f:
                        f.write(r.content)
        except Exception as e:
            print(f"Failed to download {path}: {e}")

def main():
    setup_directories()
    download_clap()
    prepare_fma_pop()
    download_mtg_jamendo()
    print("\nAll setup tasks completed.")

if __name__ == "__main__":
    main()
