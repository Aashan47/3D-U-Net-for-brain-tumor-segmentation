#!/usr/bin/env python3
"""
Data preprocessing script for BraTS dataset.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.preprocessing import BraTSPreprocessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess BraTS Dataset")
    
    # Input/Output paths
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to raw BraTS data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save preprocessed data")
    
    # Preprocessing parameters
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Target voxel spacing (mm)")
    parser.add_argument("--target_orientation", type=str, default="RAS",
                        help="Target image orientation")
    parser.add_argument("--intensity_range", type=float, nargs=2, default=[-175, 250],
                        help="Intensity clipping range")
    
    # Processing options
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--max_subjects", type=int, 
                        help="Maximum number of subjects to process (for testing)")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only run dataset validation, skip preprocessing")
    
    # Cross-validation splits
    parser.add_argument("--create_splits", action="store_true",
                        help="Create cross-validation splits")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    parser.add_argument("--stratify", action="store_true",
                        help="Stratify splits by tumor presence")
    
    return parser.parse_args()


def main():
    """Main preprocessing function."""
    args = parse_args()
    
    print("BraTS Dataset Preprocessing")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target spacing: {args.target_spacing} mm")
    print(f"Target orientation: {args.target_orientation}")
    print(f"Intensity range: {args.intensity_range}")
    print(f"Workers: {args.num_workers}")
    if args.max_subjects:
        print(f"Max subjects: {args.max_subjects}")
    print()
    
    # Initialize preprocessor
    preprocessor = BraTSPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_spacing=tuple(args.target_spacing),
        target_orientation=args.target_orientation,
        intensity_range=tuple(args.intensity_range),
        num_workers=args.num_workers,
    )
    
    # Run validation
    print("Step 1: Dataset Validation")
    print("-" * 30)
    validation_df = preprocessor.validate_dataset()
    
    valid_subjects = validation_df["valid"].sum()
    total_subjects = len(validation_df)
    
    print(f"Validation Results:")
    print(f"  Valid subjects: {valid_subjects}/{total_subjects}")
    print(f"  Success rate: {valid_subjects/total_subjects*100:.1f}%")
    
    # Show invalid subjects if any
    invalid_subjects = validation_df[~validation_df["valid"]]
    if len(invalid_subjects) > 0:
        print(f"\nInvalid subjects ({len(invalid_subjects)}):")
        for _, row in invalid_subjects.iterrows():
            print(f"  {row['subject_id']}: {row['missing_files'] + row['file_errors'] + row['shape_mismatches']}")
    
    # Exit if validation only
    if args.validate_only:
        print("\nValidation complete. Exiting (--validate_only specified).")
        return
    
    # Run preprocessing
    print("\nStep 2: Data Preprocessing")
    print("-" * 30)
    
    summary = preprocessor.preprocess_dataset(max_subjects=args.max_subjects)
    
    print(f"Preprocessing Results:")
    print(f"  Total subjects: {summary['total_subjects']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Success rate: {summary['success_rate']*100:.1f}%")
    
    # Create cross-validation splits
    if args.create_splits:
        print("\nStep 3: Creating Cross-Validation Splits")
        print("-" * 30)
        
        splits_df = preprocessor.create_dataset_split(
            n_folds=args.n_folds,
            seed=args.seed,
            stratify_by_tumor_presence=args.stratify,
        )
        
        print(f"Created {args.n_folds}-fold cross-validation splits")
        print("Subjects per fold:")
        fold_counts = splits_df["fold"].value_counts().sort_index()
        for fold, count in fold_counts.items():
            print(f"  Fold {fold}: {count} subjects")
        
        # Show stratification results if used
        if args.stratify:
            print("\nStratification by tumor presence:")
            for fold in range(args.n_folds):
                fold_subjects = splits_df[splits_df["fold"] == fold]["subject_id"].tolist()
                # This would require analyzing each subject's segmentation
                # For now, just show we created stratified splits
                print(f"  Fold {fold}: {len(fold_subjects)} subjects")
    
    print("\nPreprocessing complete!")
    print(f"Preprocessed data saved to: {args.output_dir}")
    
    # Print summary files created
    output_path = Path(args.output_dir)
    summary_files = [
        "validation_report.csv",
        "preprocessing_summary.json",
    ]
    if args.create_splits:
        summary_files.append("dataset_splits.csv")
    
    print("\nSummary files created:")
    for file in summary_files:
        file_path = output_path / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")
    
    print("\nNext steps:")
    print("1. Review validation and preprocessing reports")
    print("2. Use preprocessed data for training:")
    print(f"   python scripts/train.py --data_path {args.output_dir} --fold 0")


if __name__ == "__main__":
    main()