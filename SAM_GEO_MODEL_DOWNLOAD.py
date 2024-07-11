import requests
import os

def download_sam_checkpoint(model_type="vit_h", save_directory="model/"):
    """Downloads the SAM checkpoint file for the specified model type."""

    if model_type not in ["vit_h", "vit_l", "vit_b"]:
        raise ValueError("Invalid model_type. Choose from 'vit_h', 'vit_l', or 'vit_b'.")

    model_filename = f"sam_{model_type}_4b8939.pth"
    model_url = f"https://dl.fbaipublicfiles.com/segment_anything/{model_filename}"

    save_path = os.path.join(save_directory, model_filename)
    os.makedirs(save_directory, exist_ok=True)

    print(f"Downloading {model_filename} to {save_path}...")
    with requests.get(model_url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Download complete!")
    return save_path

download_sam_checkpoint(save_directory="model/")