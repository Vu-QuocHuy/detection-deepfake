#!/usr/bin/env python3
"""
Testing and Evaluation Script for DeepFake Detection
Comprehensive evaluation with multiple metrics.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
import pandas as pd
import os

from deepfake_detector.models import DeepFakeDetector, TriStreamDeepFakeDetector
from deepfake_detector.data import create_combined_dataset, get_val_transforms, create_dataloaders
from deepfake_detector.utils import (
    setup_logger, calculate_comprehensive_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curve
)
from sklearn.metrics import confusion_matrix, accuracy_score


def _effnet_input_size(model_name: str) -> int:
    name = model_name.lower()
    if "b0" in name:
        return 224
    if "b1" in name:
        return 240
    if "b2" in name:
        return 260
    if "b3" in name:
        return 300
    if "b4" in name:
        return 380
    if "b5" in name:
        return 456
    if "b6" in name:
        return 528
    if "b7" in name:
        return 600
    return 224


def test_model(model, dataloader, device, logger):
    """Test the model and collect predictions."""
    model.eval()

    all_probs = []
    all_labels = []
    all_image_paths = []

    logger.info("Running inference on test set...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            if len(batch) == 3:
                images, labels, image_paths = batch
            else:
                images, labels = batch
                image_paths = None
            images = images.to(device)

            outputs = model(images)
            probs = softmax(outputs.cpu().numpy(), axis=1)

            all_probs.append(probs)
            all_labels.extend(labels.numpy())
            if image_paths is not None:
                all_image_paths.extend(list(image_paths))

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)

    return all_probs, all_labels, all_image_paths


def extract_video_id(image_path: str) -> str:
    """
    Recover source video id from extracted face filename.
    Expected naming from extract_faces.py:
      <video_stem>-<timestamp>.jpg
    """
    stem = Path(image_path).stem
    if '-' in stem:
        return stem.rsplit('-', 1)[0]
    return stem


def compute_video_level_metrics(pred_df: pd.DataFrame):
    grouped = pred_df.groupby('video_id').agg(
        true_label=('true_label', 'max'),
        prob_real=('prob_real', 'mean'),
        prob_fake=('prob_fake', 'mean'),
    ).reset_index()
    grouped['pred_label'] = (grouped['prob_real'] >= 0.5).astype(int)
    video_acc = accuracy_score(grouped['true_label'].values, grouped['pred_label'].values)
    return grouped, video_acc


def main():
    parser = argparse.ArgumentParser(
        description='Test DeepFake Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--test-real', type=str, nargs='+', required=True,
                        help='Paths to test real images directories')
    parser.add_argument('--test-fake', type=str, nargs='+', required=True,
                        help='Paths to test fake images directories')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of DataLoader workers')
    parser.add_argument('--model', type=str, default='efficientnet-b1',
                        help='Model architecture')
    parser.add_argument('--multistream', action='store_true',
                        help='Use tri-stream RGB+frequency+SRM-like detector')
    parser.add_argument('--freq-model', type=str, default='efficientnet-b2',
                        help='EfficientNet variant for frequency stream')
    parser.add_argument('--srm-model', type=str, default='efficientnet-b0',
                        help='EfficientNet variant for SRM-like stream')
    parser.add_argument('--srm-filters', type=int, default=30,
                        help='Number of learnable SRM filters (noise channels)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to CSV')
    parser.add_argument('--video-level', action='store_true',
                        help='Compute video-level metrics from frame predictions')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        name='testing',
        log_file=str(output_dir / 'test.log'),
        level='INFO'
    )

    logger.info("="*60)
    logger.info("DeepFake Detection Testing")
    logger.info("="*60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create test dataset
    logger.info("Creating test dataset...")

    image_size = _effnet_input_size(args.model)
    test_transforms = get_val_transforms(image_size)

    test_real_config = [(path, -1) for path in args.test_real]
    test_fake_config = [(path, -1) for path in args.test_fake]

    test_dataset = create_combined_dataset(
        test_real_config,
        test_fake_config,
        test_transforms,
        return_paths=True
    )
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Create dataloader
    _, _, test_loader = create_dataloaders(
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    if args.multistream:
        model = TriStreamDeepFakeDetector(
            rgb_model=args.model,
            freq_model=args.freq_model,
            srm_model=args.srm_model,
            srm_filters=args.srm_filters,
            pretrained=False,
        )
    else:
        model = DeepFakeDetector(model_name=args.model, pretrained=False)
    model.load_checkpoint(args.checkpoint, device=str(device))
    model = model.to(device)
    model.eval()

    # Test model
    probs, labels, image_paths = test_model(model, test_loader, device, logger)

    # Get predictions for real class (index 0 is fake, index 1 is real)
    # Assuming model outputs [fake_prob, real_prob]
    pred_labels = np.argmax(probs, axis=1)
    real_probs = probs[:, 1]  # Probability of being real

    # Calculate metrics
    logger.info("\nCalculating metrics...")
    metrics = calculate_comprehensive_metrics(real_probs, labels, pred_labels)

    # Print metrics
    print_metrics(metrics, title="Test Results")

    # Save metrics
    metrics_file = output_dir / 'metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("Test Results\n")
        f.write("="*60 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Metrics saved to: {metrics_file}")

    # Plot confusion matrix
    conf_mat = confusion_matrix(labels, pred_labels)
    plot_confusion_matrix(
        conf_mat,
        class_names=['Fake', 'Real'],
        title='Test Set Confusion Matrix',
        save_path=str(output_dir / 'confusion_matrix.png'),
        show=False
    )
    logger.info("Confusion matrix saved")

    # Plot ROC curve
    from deepfake_detector.utils.metrics import get_EER_states
    EER, optimal_thr, FRR_list, FAR_list = get_EER_states(real_probs, labels)

    plot_roc_curve(
        FRR_list, FAR_list, EER,
        title=f'ROC Curve (EER={EER:.4f})',
        save_path=str(output_dir / 'roc_curve.png'),
        show=False
    )
    logger.info("ROC curve saved")

    # Save predictions
    if args.save_predictions:
        predictions_df = pd.DataFrame({
            'image_path': image_paths if image_paths else [None] * len(labels),
            'true_label': labels,
            'pred_label': pred_labels,
            'prob_fake': probs[:, 0],
            'prob_real': probs[:, 1]
        })
        if image_paths:
            predictions_df['video_id'] = predictions_df['image_path'].map(extract_video_id)
        predictions_file = output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to: {predictions_file}")

        if args.video_level and image_paths:
            video_df, video_acc = compute_video_level_metrics(predictions_df)
            video_file = output_dir / 'video_predictions.csv'
            video_df.to_csv(video_file, index=False)
            logger.info(f"Video-level predictions saved to: {video_file}")

            video_metrics_file = output_dir / 'video_metrics.txt'
            with open(video_metrics_file, 'w') as f:
                f.write("Video-level Results\n")
                f.write("="*60 + "\n")
                f.write(f"num_videos: {len(video_df)}\n")
                f.write(f"video_accuracy: {video_acc:.6f}\n")
            logger.info(f"Video-level metrics saved to: {video_metrics_file}")

    logger.info("\nTesting complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
