# # datasets/download_public_datasets.py

# import os
# import requests
# import zipfile
# import gdown

# DATA_DIR = "data"
# os.makedirs(DATA_DIR, exist_ok=True)

# def download_url(url, dest):
#     if os.path.exists(dest):
#         print(f"[SKIP] {dest} already exists.")
#         return
#     print(f"Downloading {url} → {dest}")
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(dest, "wb") as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)
#     print(f"[OK] Saved to {dest}")

# def extract_zip(src, dst):
#     print(f"Extracting {src} → {dst}")
#     with zipfile.ZipFile(src, "r") as z:
#         z.extractall(dst)
#     print(f"[OK] Extracted to {dst}")

# def download_fdmse_isl():
#     """Download FDMSE-ISL dataset info page—requires manual download"""
#     print("ℹ️ FDMSE-ISL dataset (2002-word ISL videos) available via PapersWithCode link.")
#     print("Please visit the link and download manually: FDMSE-ISL :contentReference[oaicite:1]{index=1}")

# def download_isl_csltr():
#     """Download ISL-CSLTR from Mendeley (images and videos dataset)"""
#     url = "https://data.mendeley.com/public-files/datasets/kcmpdxky7p/files/4a6e0be2-2a3f-4eca-9d51-590700f9764c"  # example direct link
#     dest_zip = os.path.join(DATA_DIR, "ISL-CSLTR.zip")
#     download_url(url, dest_zip)
#     extract_zip(dest_zip, os.path.join(DATA_DIR, "ISL-CSLTR"))

# def download_indian_sign_images():
#     """ISL phrase image dataset from Mendeley (44 phrases, 40 images each)"""
#     url = "https://data.mendeley.com/public-files/datasets/w7fgy7jvs8/files/3b7b7f7e-..."  # update actual link
#     dest_zip = os.path.join(DATA_DIR, "ISL_phrases.zip")
#     download_url(url, dest_zip)
#     extract_zip(dest_zip, os.path.join(DATA_DIR, "ISL_phrases"))

# def download_spreadthesign():
#     print("SpreadTheSign is an online dictionary—bulk download unavailable, refer to website :contentReference[oaicite:2]{index=2}")

# def main():
#     print("👉 Downloading ISL datasets")
#     download_fdmse_isl()
#     download_isl_csltr()
#     download_indian_sign_images()

#     print("\n👉 Reminder: Manual download required for some datasets like SpreadTheSign.")

# if __name__ == "__main__":
#     main()

# datasets/download_public_datasets.py

# import os
# import requests
# import zipfile

# DATA_DIR = "data/isl_datasets"
# os.makedirs(DATA_DIR, exist_ok=True)

# def download_url(url, dest):
#     if os.path.exists(dest):
#         print(f"[SKIP] {dest} already exists.")
#         return
#     print(f"Downloading {url} → {dest}")
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(dest, "wb") as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)
#     print(f"[OK] Saved to {dest}")

# def extract_zip(src, dst):
#     print(f"Extracting {src} → {dst}")
#     with zipfile.ZipFile(src, "r") as z:
#         z.extractall(dst)
#     print(f"[OK] Extracted to {dst}")

# def download_isl_csltr():
#     url = "https://data.mendeley.com/public-files/datasets/kcmpdxky7p/1/files/a0b1ce50-..."  # Complete direct URL from Mendeley
#     dest = os.path.join(DATA_DIR, "ISL-CSLTR.zip")
#     download_url(url, dest)
#     extract_zip(dest, os.path.join(DATA_DIR, "ISL-CSLTR"))

# def download_isl_phrases():
#     url = "https://data.mendeley.com/public-files/datasets/w7fgy7jvs8/1/files/3b7b7f7e-..."  # Direct URL needed
#     dest = os.path.join(DATA_DIR, "ISL_phrases.zip")
#     download_url(url, dest)
#     extract_zip(dest, os.path.join(DATA_DIR, "ISL_phrases"))

# def download_isl_islan():
#     url = "https://data.mendeley.com/public-files/datasets/rc349j45m5/1/files/abcd1234-..."  # Direct URL
#     dest = os.path.join(DATA_DIR, "ISLAN.zip")
#     download_url(url, dest)
#     extract_zip(dest, os.path.join(DATA_DIR, "ISLAN"))

# def download_isl_vt():
#     url = "https://data.mendeley.com/public-files/datasets/98mzk82wbb/1/files/efgh5678-..."  # Direct URL to ISLVT zip
#     dest = os.path.join(DATA_DIR, "ISLVT.zip")
#     download_url(url, dest)
#     extract_zip(dest, os.path.join(DATA_DIR, "ISLVT"))

# def download_isl_emergency():
#     url = "https://data.mendeley.com/public-files/datasets/2vfdm42337/1/files/ijkl9012-..."  # Direct URL for emergency ISL
#     dest = os.path.join(DATA_DIR, "ISL_emergency.zip")
#     download_url(url, dest)
#     extract_zip(dest, os.path.join(DATA_DIR, "ISL_emergency"))

# def main():
#     print("👉 Downloading ISL public datasets")
#     download_isl_csltr()      # ISL-CSLTR 📘 :contentReference[oaicite:1]{index=1}
#     download_isl_phrases()    # 44 ISL phrases 📘 :contentReference[oaicite:2]{index=2}
#     download_isl_islan()      # ISLAN (alphabet & numbers) 📘 :contentReference[oaicite:3]{index=3}
#     download_isl_vt()         # ISL Video/Text dataset (ISLVT) 📘 :contentReference[oaicite:4]{index=4}
#     download_isl_emergency()  # Emergency signs (8 words) 📘 :contentReference[oaicite:5]{index=5}

#     print("\n✅ All downloads initiated. Please check folder:", DATA_DIR)

# if __name__ == "__main__":
#     main()
# datasets/download_public_datasets.py  
# datasets/download_all_isl_datasets.py

# import os
# import subprocess
# import gdown

# DATA_DIR = "data/isl_datasets"
# os.makedirs(DATA_DIR, exist_ok=True)

# def download_kaggle_dataset(dataset_slug, output_dir):
#     print(f"📦 Downloading from Kaggle: {dataset_slug}")
#     subprocess.run([
#         "kaggle", "datasets", "download", "-d", dataset_slug, "-p", output_dir, "--unzip"
#     ])

# def download_gdrive_dataset(file_id, output_path):
#     if os.path.exists(output_path):
#         print(f"[SKIP] {output_path} already exists.")
#         return
#     url = f"https://drive.google.com/uc?id={file_id}"
#     gdown.download(url, output_path, quiet=False)
#     print(f"[OK] Downloaded to {output_path}")

# def main():
#     print("📥 Downloading ISL datasets...\n")

#     # ✅ Kaggle datasets
#     try:
#         download_kaggle_dataset("drblack00/isl-csltr-indian-sign-language-dataset", DATA_DIR)
#         download_kaggle_dataset("harsh0239/isl-indian-sign-language-video-dataset", DATA_DIR)
#     except Exception as e:
#         print(f"❌ Kaggle download failed. Make sure Kaggle CLI is installed and authenticated.\n{e}")

#     # ✅ Google Drive (if public mirrors are found)
#     # Add verified GDrive file IDs if available
#     gdrive_datasets = {
#         # "ISLAN": "1vEGY0ztQ2Y4J8LriGgdJSedz5YfLZGRm",   # Example only
#     }

#     for name, file_id in gdrive_datasets.items():
#         dest_path = os.path.join(DATA_DIR, f"{name}.zip")
#         download_gdrive_dataset(file_id, dest_path)

#     # ⚠️ Manual Mendeley datasets
#     print("\n📌 Please download these manually (ZIPs) and place them in `data/isl_datasets/`:")
#     print("1. ISL-CSLTR: https://data.mendeley.com/datasets/kcmpdxky7p/1")
#     print("2. ISLAN (Alphabet & Numbers): https://data.mendeley.com/datasets/rc349j45m5/1")
#     print("3. ISL Phrases: https://data.mendeley.com/datasets/w7fgy7jvs8/1")
#     print("4. ISLVT (Video-Text Pairs): https://data.mendeley.com/datasets/98mzk82wbb/1")
#     print("5. ISL Emergency Gestures: https://data.mendeley.com/datasets/2vfdm42337/1")

#     print("\n✅ Done. You can now process the downloaded datasets.")

# if __name__ == "__main__":
#     main()
# datasets/download_all_isl_datasets.py

import os
import subprocess
import gdown
import json

DATA_DIR = "data/isl_datasets"
os.makedirs(DATA_DIR, exist_ok=True)

# Step 1: Set up Kaggle credentials
def setup_kaggle_credentials():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    creds = {
        "username": "avinashkumarkashyap",
        "key": "230c3280cdca2aa6af00a8c2d60f7c48"
    }

    with open(kaggle_json_path, "w") as f:
        json.dump(creds, f)
    os.chmod(kaggle_json_path, 0o600)
    print("✅ Kaggle API credentials set up.")

# Step 2: Download from Kaggle
def download_kaggle_dataset(dataset_slug, output_dir):
    print(f"\n📦 Downloading from Kaggle: {dataset_slug}")
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset_slug, "-p", output_dir, "--unzip"
    ], check=True)

# Step 3: (Optional) Google Drive downloads via gdown
def download_gdrive_dataset(file_id, output_path):
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists.")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    print(f"[OK] Downloaded to {output_path}")

def main():
    print("📥 Downloading ISL datasets...\n")

    # Step 1: Setup Kaggle API credentials
    setup_kaggle_credentials()

    # Step 2: Kaggle datasets
    try:
        download_kaggle_dataset("drblack00/isl-csltr-indian-sign-language-dataset", DATA_DIR)
        download_kaggle_dataset("harsh0239/isl-indian-sign-language-video-dataset", DATA_DIR)
    except Exception as e:
        print(f"❌ Kaggle download failed:\n{e}")

    # Step 3: Manual Mendeley datasets
    print("\n📌 Please download these manually and place them in `data/isl_datasets/` folder:")
    print("1. ISL-CSLTR ➤ https://data.mendeley.com/datasets/kcmpdxky7p/1")
    print("2. ISLAN (A-Z, 0-9) ➤ https://data.mendeley.com/datasets/rc349j45m5/1")
    print("3. ISL Phrases ➤ https://data.mendeley.com/datasets/w7fgy7jvs8/1")
    print("4. ISLVT (Video-Text Pairs) ➤ https://data.mendeley.com/datasets/98mzk82wbb/1")
    print("5. ISL Emergency ➤ https://data.mendeley.com/datasets/2vfdm42337/1")

    print("\n✅ Download step complete. Proceed to keypoint extraction next.")

if __name__ == "__main__":
    main()
