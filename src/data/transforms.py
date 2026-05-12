"""Transform module for Conifer Family Classification.

Provides augmentation pipelines optimized for wood cross-section images.
Cross-section images are isotropic — vertical flips and 90-degree rotations
are semantically valid augmentations that preserve taxonomic features.
"""

from torchvision import transforms


def get_train_transforms(
    img_size: int = 224,
    use_90rotation: bool = False,
    degrees: float = 30.0,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> transforms.Compose:
    """Get training augmentation pipeline.

    Augmentations are chosen with the isotropic nature of wood cross-section
    images in mind: resin canals, tracheids, and ray cells have no canonical
    orientation, so rotations and flips preserve biological validity.

    Args:
        img_size: Target image size (default 224)
        use_90rotation: Use discrete 90-degree rotation instead of continuous.
            Recommended: True for fine-tuning (larger datasets), False for
            small datasets where heavy augmentation could hurt.
        degrees: Max rotation angle for continuous rotation.
        hflip_prob: Probability of horizontal flip.
        vflip_prob: Probability of vertical flip.
        brightness: ColorJitter brightness range.
        contrast: ColorJitter contrast range.

    Returns:
        Composed training transforms
    """
    aug_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=hflip_prob),
        transforms.RandomVerticalFlip(p=vflip_prob),
    ]

    if use_90rotation:
        aug_list.append(transforms.RandomChoice([
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 0)),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 90)),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 180)),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 270)),
        ]))
    else:
        aug_list.append(transforms.RandomRotation(degrees=degrees))

    aug_list.extend([
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(aug_list)


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Get validation/test transform (no augmentation).

    Args:
        img_size: Target image size

    Returns:
        Composed validation transforms
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
