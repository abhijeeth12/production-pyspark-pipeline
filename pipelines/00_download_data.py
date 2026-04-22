#!/usr/bin/env python3
"""
Download the 2015 US Flight Delays dataset from a public source.
Dataset: 2015 Flight Delays and Cancellations (BTS via Kaggle mirror)
"""
import os
import urllib.request
import zipfile
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, dest_path, desc=""):
    print(f"Downloading {desc}...")
    def reporthook(blocknum, blocksize, totalsize):
        downloaded = blocknum * blocksize
        if totalsize > 0:
            pct = min(downloaded * 100 / totalsize, 100)
            bar = int(pct / 5)
            sys.stdout.write(f"\r  [{'#'*bar}{' '*(20-bar)}] {pct:.1f}% ({downloaded/1e6:.1f}/{totalsize/1e6:.1f} MB)")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print(f"\n  Saved to {dest_path}")

# ---- Primary source: Use the Airline On-Time Performance data from BTS ----
# We'll use the 'airlines' dataset from OpenML (publicly available, ~500K rows)
# which is a well-known benchmark for delay prediction.
# For a large dataset we'll use the full 2015 airlines CSV from an open repository.

URLS = [
    # Attempt 1: Large airlines dataset from GitHub releases / open data
    ("https://figshare.com/ndownloader/files/3888791", "2008_flights.csv.bz2", "2008 Flight Data (7M rows)"),
]

# Actually, let's use the UCI airlines dataset + generate an extended version
# Better approach: download from a reliable open-data source
# We'll use the 2009-2018 flight data available from BTS open data portal

import urllib.request

def download_bts_data():
    """Download BTS airline on-time data for 2015."""
    # BTS has monthly zip files for on-time performance
    base_url = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    
    months_to_download = [1, 2, 3]  # Q1 2015 = ~1.5M rows
    year = 2015
    
    downloaded_csvs = []
    for month in months_to_download:
        url = base_url.format(year=year, month=month)
        zip_path = os.path.join(DATA_DIR, f"bts_{year}_{month:02d}.zip")
        csv_name = f"flights_{year}_{month:02d}.csv"
        csv_path = os.path.join(DATA_DIR, csv_name)
        
        if os.path.exists(csv_path):
            print(f"  {csv_name} already exists, skipping download.")
            downloaded_csvs.append(csv_path)
            continue
        
        try:
            print(f"\nDownloading BTS data: {year}-{month:02d}...")
            download_file(url, zip_path, f"BTS {year}-{month:02d}")
            
            print(f"  Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                print(f"  Files in zip: {names}")
                for name in names:
                    if name.endswith('.csv') and 'On_Time' in name:
                        zf.extract(name, DATA_DIR)
                        extracted = os.path.join(DATA_DIR, name)
                        os.rename(extracted, csv_path)
                        print(f"  Extracted to {csv_path}")
                        break
            os.remove(zip_path)
            downloaded_csvs.append(csv_path)
        except Exception as e:
            print(f"  Error: {e}")
    
    return downloaded_csvs

if __name__ == '__main__':
    csvs = download_bts_data()
    print(f"\nDownloaded {len(csvs)} files:")
    for f in csvs:
        size = os.path.getsize(f) if os.path.exists(f) else 0
        print(f"  {f}: {size/1e6:.1f} MB")
    print("\nData download complete!")
