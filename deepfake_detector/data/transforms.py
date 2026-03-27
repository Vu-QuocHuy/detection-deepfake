"""
Data augmentation and preprocessing transforms.
Uses albumentations for efficient image transformations.

Augmentation levels:
  light  (default) : flip + crop — fast, baseline
  medium           : + compression + blur + color — cross-dataset ↑
  heavy            : full pipeline — maximum generalization
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


def get_train_transforms(
    image_size: int = 224,
    use_heavy_augmentation: bool = False,
    augmentation_level: str = 'medium',
) -> A.Compose:
    """
    Get training data augmentation pipeline.

    Args:
        image_size: Target image size.
        use_heavy_augmentation: Legacy flag — sets augmentation_level='heavy'.
        augmentation_level: 'light' | 'medium' | 'heavy'
            - light  : flip + crop only (fast training)
            - medium : + JPEG compression (30-90) + blur + color jitter
                       Strongly recommended for cross-dataset generalization.
            - heavy  : full pipeline with distortions + cutout

    Returns:
        Albumentations Compose with training transforms.
    """
    if use_heavy_augmentation:
        augmentation_level = 'heavy'

    _norm = [
        A.Resize(image_size, image_size, always_apply=True),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    always_apply=True),
        ToTensorV2(),
    ]

    # ── Light ──────────────────────────────────────────────────────────────────
    if augmentation_level == 'light':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(image_size, image_size, scale=(0.7, 1.0), p=0.5),
        ] + _norm)

    # ── Medium (DEFAULT — best for cross-dataset) ──────────────────────────────
    # Key insight: Celeb-DF has stronger JPEG compression than FF++ C23.
    # Simulating aggressive compression during training is the single most
    # effective augmentation for cross-dataset generalization.
    if augmentation_level == 'medium':
        return A.Compose([
            # Geometry
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(image_size, image_size,
                                scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.5),
            A.Rotate(limit=10, p=0.3),

            # Compression + blur — critical for cross-dataset robustness
            A.OneOf([
                A.ImageCompression(quality_lower=30, quality_upper=90, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0),
            ], p=0.5),

            # Noise
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),

            # Color jitter — simulates different cameras/lighting
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.05, p=0.4),
        ] + _norm)

    # ── Heavy ──────────────────────────────────────────────────────────────────
    return A.Compose([
        # Geometry
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.05),
        A.RandomResizedCrop(image_size, image_size,
                            scale=(0.5, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=15, p=0.3),

        # Distortions
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1, p=1.0),
            A.GridDistortion(p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        ], p=0.2),

        # Compression + degradation (most important for cross-dataset)
        A.OneOf([
            A.ImageCompression(quality_lower=30, quality_upper=90, p=1.0),
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.Downscale(scale_min=0.4, scale_max=0.9, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.6),

        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

        # Color
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                             val_shift_limit=20, p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                   b_shift_limit=15, p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.CLAHE(p=0.2),

        # Cutout
        A.CoarseDropout(max_holes=8, max_height=image_size // 8,
                        max_width=image_size // 8, fill_value=0, p=0.3),
    ] + _norm)


def get_val_transforms(image_size: int = 224) -> Compose:
    """
    Get validation/test data preprocessing pipeline.

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose object with validation transforms

    Example:
        >>> transforms = get_val_transforms(224)
        >>> preprocessed = transforms(image=image)
    """
    return Compose([
        Resize(image_size, image_size, always_apply=True),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            always_apply=True
        ),
        ToTensorV2()
    ])


def get_test_time_augmentation_transforms(image_size: int = 224) -> list:
    """
    Get multiple transform variants for test-time augmentation.

    Args:
        image_size: Target image size

    Returns:
        List of transform compositions for TTA

    Example:
        >>> tta_transforms = get_test_time_augmentation_transforms(224)
        >>> predictions = [model(transform(image=img)['image']) for transform in tta_transforms]
    """
    return [
        # Original
        get_val_transforms(image_size),

        # Horizontal flip
        Compose([
            HorizontalFlip(p=1.0),
            Resize(image_size, image_size, always_apply=True),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True
            ),
            ToTensorV2()
        ]),

        # Slight brightness adjustment
        Compose([
            RandomBrightness(limit=0.1, p=1.0),
            Resize(image_size, image_size, always_apply=True),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True
            ),
            ToTensorV2()
        ]),

        # Slight contrast adjustment
        Compose([
            RandomContrast(limit=0.1, p=1.0),
            Resize(image_size, image_size, always_apply=True),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True
            ),
            ToTensorV2()
        ]),
    ]
