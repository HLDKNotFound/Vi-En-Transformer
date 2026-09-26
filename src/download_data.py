import os
import argparse
from huggingface_hub import hf_hub_download

REPO_ID = "ncduy/mt-en-vi"
REVISION = "refs/convert/parquet"

FILES = {
    "train_0.parquet": "default/train/0000.parquet",
    "train_1.parquet": "default/train/0001.parquet",
    "val.parquet": "default/validation/0000.parquet",
    "test.parquet": "default/test/0000.parquet",
}

def download_dataset(target_dir="data/raw"):
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading dataset from {REPO_ID} (revision: {REVISION}) into {target_dir}...")
    
    for local_name, hf_path in FILES.items():
        dest_path = os.path.join(target_dir, local_name)
        if os.path.exists(dest_path):
            print(f"  [✓] {local_name} already exists ({os.path.getsize(dest_path) / (1024*1024):.2f} MB), skipping.")
            continue
        
        print(f"  Downloading {hf_path} -> {dest_path}...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            revision=REVISION
        )
        # Copy or symlink to final destination
        import shutil
        shutil.copyfile(downloaded, dest_path)
        print(f"  [✓] Saved {local_name} ({os.path.getsize(dest_path) / (1024*1024):.2f} MB)")
        
    print("Dataset download complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download parquet dataset for Vietnamese-English MT")
    parser.add_argument("--target_dir", default="data/raw", help="Target directory for raw parquet files")
    args = parser.parse_args()
    
    download_dataset(args.target_dir)
