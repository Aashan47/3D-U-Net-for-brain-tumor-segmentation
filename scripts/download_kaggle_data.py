#!/usr/bin/env python3
"""
Download and prepare BraTS 2020 dataset from Kaggle for training.
"""

import os
import sys
import subprocess
from pathlib import Path
import zipfile
import shutil

def install_requirements():
    """Install required packages for Kaggle data download."""
    print("Installing required packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "kaggle", "kagglehub", "pandas"
    ])

def setup_kaggle_credentials():
    """Setup Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    print("Setting up Kaggle credentials...")
    print("Please ensure you have:")
    print("1. Downloaded kaggle.json from your Kaggle account")
    print("2. Placed it in ~/.kaggle/kaggle.json")
    print("3. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        print(f"\nERROR: Kaggle credentials not found at {kaggle_json}")
        print("Please download kaggle.json from https://www.kaggle.com/settings")
        print("and place it in ~/.kaggle/kaggle.json")
        return False
    
    # Set correct permissions
    os.chmod(kaggle_json, 0o600)
    return True

def download_brats_dataset(data_dir):
    """Download BraTS 2020 dataset from Kaggle."""
    print(f"Downloading BraTS 2020 dataset to {data_dir}...")
    
    # Create data directory
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download using Kaggle API
    try:
        cmd = [
            "kaggle", "datasets", "download", 
            "awsaf49/brats2020-training-data",
            "-p", str(data_dir),
            "--unzip"
        ]
        subprocess.check_call(cmd)
        print("✓ Dataset downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return False

def organize_dataset(data_dir):
    """Organize the downloaded dataset into expected structure."""
    data_dir = Path(data_dir)
    
    print("Organizing dataset structure...")
    
    # Look for the extracted data
    training_dir = None
    for item in data_dir.iterdir():
        if item.is_dir() and "brats" in item.name.lower():
            training_dir = item
            break
    
    if not training_dir:
        print("Looking for training data in subdirectories...")
        for item in data_dir.rglob("*"):
            if item.is_dir() and any(x in item.name.lower() for x in ["training", "brats", "data"]):
                training_dir = item
                break
    
    if training_dir:
        print(f"Found training data at: {training_dir}")
        
        # Count subjects
        subjects = list(training_dir.glob("BraTS*"))
        print(f"Found {len(subjects)} subjects")
        
        # Show structure
        if subjects:
            sample_subject = subjects[0]
            print(f"\nSample subject structure ({sample_subject.name}):")
            for file in sample_subject.glob("*.nii.gz"):
                print(f"  - {file.name}")
        
        return str(training_dir)
    else:
        print("Could not find training data directory")
        return None

def verify_dataset(data_path):
    """Verify the dataset is properly organized."""
    data_path = Path(data_path)
    
    print(f"Verifying dataset at {data_path}...")
    
    # Check for subject directories
    subjects = list(data_path.glob("BraTS*"))
    print(f"Found {len(subjects)} subjects")
    
    if len(subjects) == 0:
        print("ERROR: No BraTS subject directories found!")
        return False
    
    # Check first few subjects for required modalities
    required_modalities = ["t1", "t1ce", "t2", "flair", "seg"]
    
    for i, subject in enumerate(subjects[:3]):
        print(f"\nChecking {subject.name}:")
        found_modalities = []
        
        for file in subject.glob("*.nii.gz"):
            file_name = file.name.lower()
            for mod in required_modalities:
                if mod in file_name:
                    found_modalities.append(mod)
                    print(f"  ✓ {mod}: {file.name}")
        
        missing = set(required_modalities) - set(found_modalities)
        if missing:
            print(f"  ⚠ Missing: {missing}")
    
    print(f"\n✓ Dataset verification complete!")
    print(f"Ready for training with {len(subjects)} subjects")
    return True

def main():
    """Main function to download and prepare dataset."""
    # Set paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    
    print("BraTS 2020 Kaggle Dataset Downloader")
    print("====================================")
    
    # Install requirements
    install_requirements()
    
    # Setup Kaggle credentials
    if not setup_kaggle_credentials():
        return 1
    
    # Download dataset
    if not download_brats_dataset(data_dir):
        return 1
    
    # Organize dataset
    organized_path = organize_dataset(data_dir)
    if not organized_path:
        return 1
    
    # Verify dataset
    if not verify_dataset(organized_path):
        return 1
    
    # Update launch script with correct path
    launch_script = project_root / "launch_training.sh"
    if launch_script.exists():
        print(f"\nUpdating launch script with data path: {organized_path}")
        
        # Read current script
        content = launch_script.read_text()
        
        # Update DATA_PATH line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'DATA_PATH=' in line and '/path/to/your/brats/data' in line:
                lines[i] = f'    DATA_PATH="{organized_path}"'
                break
        
        # Write back
        launch_script.write_text('\n'.join(lines))
        print("✓ Launch script updated!")
    
    print("\n🎉 Dataset preparation complete!")
    print(f"Data location: {organized_path}")
    print(f"Subjects: {len(list(Path(organized_path).glob('BraTS*')))}")
    print("\nNext steps:")
    print("1. Run: ./launch_training.sh")
    print("2. Monitor: ./check_training.sh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())