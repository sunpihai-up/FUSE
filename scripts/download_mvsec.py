import os
import gdown
import argparse
from concurrent.futures import ThreadPoolExecutor

outdoor_day_urls = {
    "outdoor_day_calib.zip": "https://drive.google.com/uc?id=1Y0sPP0ebX_cEKUCVVhJLej9TZgCxS3ME",
    "outdoor_day1_data.hdf5": "https://drive.google.com/uc?id=1JLIrw2L24zIQBmqaWvef7G2t9tsMY3H0",
    "outdoor_day1_gt.hdf5": "https://drive.google.com/uc?id=1wzUmTBxQ5wtSpB0KBogliB2IGTrCtJ7e",
    "outdoor_day2_data.hdf5": "https://drive.google.com/uc?id=1fu9GhjYcET00mMN-YbAp3eBK1YMCd3Ox",
    "outdoor_day2_gt.hdf5": "https://drive.google.com/uc?id=1zWOA92-Bw4xz1y5CzIROXWFymTFFwBBH",
}

outdoor_night_urls = {
    "outdoor_night_calib.zip": "https://drive.google.com/uc?id=1NGUBQ8b41b9murJualNeaO7M3nrIPjfm",
    "outdoor_night1_data.hdf5": "https://drive.google.com/uc?id=1giLg7JmHOGEHxRch0C-loQWdj_ddPITV",
    "outdoor_night1_gt.hdf5": "https://drive.google.com/uc?id=1RioFWB4wOV8prC4PAj0sgIJQkVUHqlQD",
    "outdoor_night2_data.hdf5": "https://drive.google.com/uc?id=1rRvRC5ZG-autd9NFV9qyB1pwsJtPX0G2",
    "outdoor_night2_gt.hdf5": "https://drive.google.com/uc?id=1-UAt4ZGIJ5JyxAd72AqXWbGoeJyylRoo",
    "outdoor_night3_data.hdf5": "https://drive.google.com/uc?id=1ADVyZmczdaEBBeCo2ebwt5nvx4-ZZf8L",
    "outdoor_night3_gt.hdf5": "https://drive.google.com/uc?id=1eMVz4BziJMBobmgsA0izEJrvDmSd-MDy",
}

def download_file(name, url, output_dir):
    """
    Download a single file from a Google Drive URL to a specified directory.

    Args:
        name (str): Name of the file to save.
        url (str): Google Drive file URL.
        output_dir (str): Directory where the file will be saved.
    """
    output = os.path.join(output_dir, name)

    print(f"Downloading {name} from: {url}")

    try:
        gdown.download(url, output, quiet=False)
    except Exception as e:
        print(f"Error downloading {name}: {e}")

def download_files(urls_dict, output_dir, max_workers=4):
    """
    Download files from a dictionary of Google Drive URLs to a specified directory using multi-threading.

    Args:
        urls_dict (dict): Dictionary where keys are file names and values are Google Drive URLs.
        output_dir (str): Directory where files will be saved.
        max_workers (int): Maximum number of threads to use for downloading.
    """
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda item: download_file(item[0], item[1], output_dir), urls_dict.items())
        
def get_args():
    parser = argparse.ArgumentParser(description='Download files from google drive')
    
    parser.add_argument('--data_root', type=str, help='Data storage directory')
    parser.add_argument('--max_workers', default=4, type=int, help='Number of data download threads')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # outdoor_day_dir = os.path.join(args.data_root, 'outdoor_day')
    # outdoor_night_dir = os.path.join(args.data_root, 'outdoor_night')
    
    download_files(outdoor_day_urls, args.data_root)
    download_files(outdoor_night_urls, args.data_root)
