import requests
from tqdm import tqdm
import zipfile


def download_data(file_id, output_path):
    url = f"https://ndownloader.figshare.com/files/{file_id}"

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))

        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading from {url}") as pbar:
            with output_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    print(f"Saved ZIP.")


def extract_zip_into_data(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print("Extraction complete.")


def delete_zip(zip_path):
    if zip_path.exists():
        zip_path.unlink()
        print(f"Deleted ZIP.")


def setup_dataset(file_id, project_root):
    extracted_folder = project_root / "ieeg_ieds_bids"
    zip_path = project_root / "ieeg_ieds_bids_final.zip"

    # Check if data already exists
    if extracted_folder.exists():
        print(f"Dataset already available {project_root}")
        return extracted_folder

    # Download ZIP if missing
    if not zip_path.exists():
        download_data(file_id, zip_path)

    # Extract ZIP directly into /data/
    extract_zip_into_data(zip_path, project_root)

    # Delete ZIP
    delete_zip(zip_path)

    print(f"Dataset extracted at {project_root}")
    return extracted_folder
