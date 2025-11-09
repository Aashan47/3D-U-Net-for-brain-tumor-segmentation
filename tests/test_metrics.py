import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from metrics.dice import DiceMetric, BraTSDiceMetric, compute_dice_score, compute_brats_dice
from metrics.hausdorff import HausdorffDistanceMetric, compute_hausdorff_distance_95
from metrics.brats_metrics import BraTSMetrics, BraTSEvaluator


class TestDiceMetric:
    """Test Dice coefficient calculations."""
    
    def test_perfect_overlap(self):
        """Test Dice score for perfect overlap."""
        # Create identical binary masks
        pred = torch.ones((1, 1, 10, 10, 10))
        target = torch.ones((1, 1, 10, 10, 10))
        
        dice_metric = DiceMetric()
        dice_score = dice_metric(pred, target)
        
        assert torch.isclose(dice_score, torch.tensor(1.0), atol=1e-6)
    
    def test_no_overlap(self):
        """Test Dice score for no overlap."""
        # Create non-overlapping masks
        pred = torch.zeros((1, 1, 10, 10, 10))
        pred[0, 0, :5, :5, :5] = 1
        
        target = torch.zeros((1, 1, 10, 10, 10))
        target[0, 0, 5:, 5:, 5:] = 1
        
        dice_metric = DiceMetric()
        dice_score = dice_metric(pred, target)
        
        assert torch.isclose(dice_score, torch.tensor(0.0), atol=1e-6)
    
    def test_partial_overlap(self):
        """Test Dice score for partial overlap."""
        # Create partially overlapping masks
        pred = torch.zeros((1, 1, 10, 10, 10))
        pred[0, 0, :7, :7, :7] = 1  # Volume = 343
        
        target = torch.zeros((1, 1, 10, 10, 10))
        target[0, 0, 3:, 3:, 3:] = 1  # Volume = 343
        
        # Intersection = 4^3 = 64
        # Union = 343 + 343 = 686
        # Dice = 2 * 64 / 686 ≈ 0.186
        
        dice_metric = DiceMetric()
        dice_score = dice_metric(pred, target)
        
        expected_dice = 2 * 64 / 686
        assert torch.isclose(dice_score, torch.tensor(expected_dice), atol=1e-3)
    
    def test_multiclass_dice(self):
        """Test multi-class Dice calculation."""
        # Create multi-class predictions
        pred = torch.zeros((1, 3, 8, 8, 8))
        pred[0, 0, :4, :4, :4] = 1  # Class 0
        pred[0, 1, 4:, :4, :4] = 1  # Class 1
        pred[0, 2, :4, 4:, :4] = 1  # Class 2
        
        target = torch.zeros((1, 3, 8, 8, 8))
        target[0, 0, :4, :4, :4] = 1  # Perfect overlap for class 0
        target[0, 1, 4:, :4, :4] = 1  # Perfect overlap for class 1
        target[0, 2, :4, 4:, :4] = 1  # Perfect overlap for class 2
        
        dice_metric = DiceMetric()
        dice_score = dice_metric(pred, target)
        
        assert torch.isclose(dice_score, torch.tensor(1.0), atol=1e-6)


class TestBraTSDiceMetric:
    """Test BraTS-specific Dice calculations."""
    
    def test_brats_regions(self):
        """Test BraTS tumor region calculations."""
        # Create mock BraTS segmentation
        pred = torch.zeros((1, 16, 16, 16))
        target = torch.zeros((1, 16, 16, 16))
        
        # Add tumor regions
        # WT = labels 1, 2, 4
        # TC = labels 1, 4
        # ET = label 4
        
        pred[0, 4:8, 4:8, 4:8] = 1    # NCR
        pred[0, 8:12, 4:8, 4:8] = 2   # ED
        pred[0, 6:8, 6:8, 6:8] = 4    # ET (within NCR)
        
        target[0, 4:8, 4:8, 4:8] = 1  # NCR
        target[0, 8:12, 4:8, 4:8] = 2 # ED
        target[0, 6:8, 6:8, 6:8] = 4  # ET
        
        brats_dice = BraTSDiceMetric()
        dice_scores = brats_dice(pred, target)
        
        # Perfect overlap should give Dice = 1.0 for all regions
        assert torch.isclose(dice_scores["WT"], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(dice_scores["TC"], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(dice_scores["ET"], torch.tensor(1.0), atol=1e-6)
    
    def test_numpy_dice_calculation(self):
        """Test numpy-based Dice calculation."""
        # Create binary masks
        pred = np.ones((20, 20, 20), dtype=bool)
        target = np.ones((20, 20, 20), dtype=bool)
        
        dice_score = compute_dice_score(pred, target)
        assert abs(dice_score - 1.0) < 1e-6
        
        # Test no overlap
        pred = np.zeros((20, 20, 20), dtype=bool)
        target = np.ones((20, 20, 20), dtype=bool)
        
        dice_score = compute_dice_score(pred, target)
        assert abs(dice_score - 0.0) < 1e-6
    
    def test_brats_numpy_dice(self):
        """Test BraTS Dice calculation with numpy."""
        # Create BraTS-style segmentation
        pred = np.zeros((30, 30, 30), dtype=np.uint8)
        target = np.zeros((30, 30, 30), dtype=np.uint8)
        
        # Add regions
        pred[10:20, 10:20, 10:20] = 1  # NCR
        pred[15:25, 10:20, 10:20] = 2  # ED
        pred[12:15, 12:15, 12:15] = 4  # ET
        
        target[10:20, 10:20, 10:20] = 1  # NCR
        target[15:25, 10:20, 10:20] = 2  # ED
        target[12:15, 12:15, 12:15] = 4  # ET
        
        dice_scores = compute_brats_dice(pred, target)
        
        # Should have perfect scores
        assert abs(dice_scores["WT"] - 1.0) < 1e-6
        assert abs(dice_scores["TC"] - 1.0) < 1e-6
        assert abs(dice_scores["ET"] - 1.0) < 1e-6


class TestHausdorffDistance:
    """Test Hausdorff distance calculations."""
    
    def test_identical_masks(self):
        """Test HD95 for identical masks."""
        # Create identical binary masks
        mask = torch.zeros((1, 10, 10, 10))
        mask[0, 3:7, 3:7, 3:7] = 1
        
        hd_metric = HausdorffDistanceMetric(percentile=95.0, spacing=(1.0, 1.0, 1.0))
        hd_distance = hd_metric(mask, mask)
        
        assert torch.isclose(hd_distance, torch.tensor(0.0), atol=1e-6)
    
    def test_empty_masks(self):
        """Test HD95 for empty masks."""
        # Both masks empty
        mask1 = torch.zeros((1, 10, 10, 10))
        mask2 = torch.zeros((1, 10, 10, 10))
        
        hd_metric = HausdorffDistanceMetric()
        hd_distance = hd_metric(mask1, mask2)
        
        assert torch.isclose(hd_distance, torch.tensor(0.0), atol=1e-6)
    
    def test_one_empty_mask(self):
        """Test HD95 when one mask is empty."""
        mask1 = torch.zeros((1, 10, 10, 10))
        mask1[0, 3:7, 3:7, 3:7] = 1
        
        mask2 = torch.zeros((1, 10, 10, 10))  # Empty
        
        hd_metric = HausdorffDistanceMetric()
        hd_distance = hd_metric(mask1, mask2)
        
        # Should return infinity for empty predictions
        assert torch.isinf(hd_distance)
    
    def test_numpy_hd95(self):
        """Test numpy HD95 calculation."""
        # Create simple test case
        pred = np.zeros((20, 20, 20), dtype=bool)
        target = np.zeros((20, 20, 20), dtype=bool)
        
        pred[8:12, 8:12, 8:12] = True
        target[10:14, 10:14, 10:14] = True  # Shifted by 2 voxels
        
        hd95 = compute_hausdorff_distance_95(pred, target, spacing=(1.0, 1.0, 1.0))
        
        # HD95 should be approximately sqrt(3) * 2 = 3.46 for this case
        assert hd95 > 0
        assert not np.isinf(hd95)


class TestBraTSMetrics:
    """Test comprehensive BraTS metrics."""
    
    def test_complete_evaluation(self):
        """Test complete BraTS evaluation pipeline."""
        # Create mock predictions and targets
        pred = torch.zeros((2, 4, 32, 32, 32))  # Batch of 2
        target = torch.zeros((2, 32, 32, 32))
        
        # Add tumor regions for first sample
        pred[0, 0, 10:20, 10:20, 10:20] = 0.1  # Background
        pred[0, 1, 10:20, 10:20, 10:20] = 0.8  # NCR
        pred[0, 2, 15:25, 10:20, 10:20] = 0.7  # ED
        pred[0, 3, 12:15, 12:15, 12:15] = 0.9  # ET
        
        target[0, 10:20, 10:20, 10:20] = 1  # NCR
        target[0, 15:25, 10:20, 10:20] = 2  # ED
        target[0, 12:15, 12:15, 12:15] = 4  # ET
        
        # Second sample with different pattern
        pred[1, 0, 5:15, 5:15, 5:15] = 0.2
        pred[1, 2, 5:15, 5:15, 5:15] = 0.8   # ED only
        
        target[1, 5:15, 5:15, 5:15] = 2      # ED only
        
        # Test metrics calculation
        metrics = BraTSMetrics(spacing=(1.0, 1.0, 1.0))
        results = metrics(pred, target)
        
        # Should have results for all regions
        assert "WT" in results
        assert "TC" in results
        assert "ET" in results
        
        # Each region should have dice score
        assert "dice" in results["WT"]
        assert "dice" in results["TC"]
        assert "dice" in results["ET"]
        
        # Dice scores should be between 0 and 1
        assert 0 <= results["WT"]["dice"] <= 1
        assert 0 <= results["TC"]["dice"] <= 1
        assert 0 <= results["ET"]["dice"] <= 1


class TestBraTSEvaluator:
    """Test BraTS evaluator class."""
    
    def test_evaluator_single_subject(self):
        """Test evaluating a single subject."""
        evaluator = BraTSEvaluator(spacing=(1.0, 1.0, 1.0))
        
        # Create mock data
        pred = np.zeros((40, 40, 40), dtype=np.uint8)
        target = np.zeros((40, 40, 40), dtype=np.uint8)
        
        # Add some tumor regions
        pred[15:25, 15:25, 15:25] = 1   # NCR
        target[15:25, 15:25, 15:25] = 1 # NCR (perfect match)
        
        pred[20:30, 15:25, 15:25] = 2   # ED
        target[22:32, 15:25, 15:25] = 2 # ED (slightly different)
        
        # Evaluate
        results = evaluator.evaluate_subject(pred, target, "test_subject", fold=0)
        
        # Check results structure
        assert "WT" in results
        assert "TC" in results
        assert "ET" in results
        
        # Check that metrics exist
        for region in results.values():
            assert "dice" in region
    
    def test_evaluator_summary_statistics(self):
        """Test summary statistics calculation."""
        evaluator = BraTSEvaluator(spacing=(1.0, 1.0, 1.0))
        
        # Evaluate multiple subjects
        for i in range(3):
            pred = np.random.randint(0, 5, (30, 30, 30)).astype(np.uint8)
            target = np.random.randint(0, 5, (30, 30, 30)).astype(np.uint8)
            
            evaluator.evaluate_subject(pred, target, f"subject_{i}", fold=0)
        
        # Get summary statistics
        summary = evaluator.get_summary_statistics()
        
        assert "WT" in summary
        assert "TC" in summary
        assert "ET" in summary
        
        # Check that mean and std are computed
        for region_stats in summary.values():
            assert any("dice_mean" in key for key in region_stats.keys())
            assert any("dice_std" in key for key in region_stats.keys())
    
    def test_evaluator_dataframe_output(self):
        """Test DataFrame output format."""
        evaluator = BraTSEvaluator(spacing=(1.0, 1.0, 1.0))
        
        # Add some evaluations
        for i in range(2):
            pred = np.random.randint(0, 5, (20, 20, 20)).astype(np.uint8)
            target = np.random.randint(0, 5, (20, 20, 20)).astype(np.uint8)
            
            evaluator.evaluate_subject(pred, target, f"subject_{i}", fold=i)
        
        # Get DataFrame
        df = evaluator.get_results_dataframe()
        
        assert len(df) == 6  # 2 subjects × 3 regions
        assert "subject_id" in df.columns
        assert "region" in df.columns
        assert "fold" in df.columns
        assert "dice" in df.columns


def test_metric_edge_cases():
    """Test edge cases for metrics."""
    # Test with very small objects
    pred_small = torch.zeros((1, 1, 10, 10, 10))
    pred_small[0, 0, 4:6, 4:6, 4:6] = 1  # 2x2x2 object
    
    target_small = torch.zeros((1, 1, 10, 10, 10))
    target_small[0, 0, 5:7, 5:7, 5:7] = 1  # 2x2x2 object, shifted
    
    dice_metric = DiceMetric()
    dice_score = dice_metric(pred_small, target_small)
    
    # Should handle small objects without errors
    assert torch.isfinite(dice_score)
    assert 0 <= dice_score <= 1


def test_batch_processing():
    """Test metrics with different batch sizes."""
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        pred = torch.rand((batch_size, 4, 16, 16, 16))
        target = torch.randint(0, 4, (batch_size, 16, 16, 16))
        
        metrics = BraTSMetrics()
        results = metrics(pred, target)
        
        # Should work with any batch size
        assert "WT" in results
        assert "TC" in results
        assert "ET" in results


if __name__ == "__main__":
    pytest.main([__file__])