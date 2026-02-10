from src.samplers.uncertainty_sampling import UncertaintySampler
from src.samplers.random_sampling import RandomSampler
from src.models.model_handler import YOLOModelHandler
from pathlib import Path
import logging
import shutil
import pandas as pd
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_csv_rows(csv_path: Path) -> int:
    """
    Count the number of data rows in a CSV file.

    This function reads a CSV file and counts all rows except the header line.
    It is primarily used to determine dataset size during active learning iterations.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file

    Returns
    -------
    int
        Number of data rows (excluding the header)
    """
    with csv_path.open() as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for header

def move_data_random(sampler: RandomSampler, train_csv: str, unlabeled_yolo_img_dir: str, unlabeled_label_dir: str, labeled_yolo_img_dir: str, labeled_label_dir: str, split_ptg: float):
    """
    Move a subset of unlabeled samples to the labeled pool randomly.

    This function selects a random subset of images from the unlabeled dataset
    and moves their corresponding images and label files into the labeled dataset directories.
    The number of moved samples is determined by the product of the current training CSV size
    and a user-specified split ratio.

    Parameters
    ----------
    sampler : RandomSampler
        The random sampling interface used for selecting samples
    train_csv : str
        Path to the current training CSV file, used to estimate dataset size
    unlabeled_yolo_img_dir : str
        Directory containing unlabeled images
    unlabeled_label_dir : str 
        Directory containing pseudo-label files for the unlabeled images
    labeled_yolo_img_dir : str 
        Destination directory for labeled images
    labeled_label_dir : str 
        Destination directory for labeled label files
    split_ptg : float
        Fraction of the total training set size that determines how many
        samples are moved from unlabeled to labeled

    Notes
    -----
    - Files are physically moved using `shutil.move`.
    - Missing files are skipped with a warning.
    - The move count is capped by the number of available unlabeled images.

    Raises
    ------
    FileNotFoundError
        If any expected file is missing during the move operation.
    """
    scores = sampler.random_sampling(unlabeled_yolo_img_dir)
    train_csv = Path(train_csv)
    unlabeled_yolo_img_dir = Path(unlabeled_yolo_img_dir)
    unlabeled_label_dir = Path(unlabeled_label_dir)
    labeled_yolo_img_dir = Path(labeled_yolo_img_dir)
    labeled_label_dir = Path(labeled_label_dir)
    total_csv_rows = count_csv_rows(train_csv)
    total_unlabeled_images = len(list(unlabeled_yolo_img_dir.glob('*.jpg')))
    move_count = min(total_unlabeled_images, int(split_ptg * total_csv_rows))
    selected_samples = [Path(s[0]) for s in scores[:move_count]]

    for img_file in selected_samples:
        label_file = img_file.with_suffix('.txt')
        src_img_yolo = unlabeled_yolo_img_dir / img_file.name
        src_lbl = unlabeled_label_dir / label_file.name
        dst_img_yolo = labeled_yolo_img_dir / img_file.name
        dst_lbl = labeled_label_dir / label_file.name

        try:
            shutil.move(src_img_yolo, dst_img_yolo)
            shutil.move(src_lbl, dst_lbl)
            logging.info(f'Moved {img_file.name} and {label_file.name} to labeled image/labels directories.')
        except FileNotFoundError as e:
            logging.warning(f'Skipping missing file: {e}')

def move_data_uncertainty(sampler: UncertaintySampler, train_csv: str, unlabeled_yolo_img_dir: str, unlabeled_label_dir: str, labeled_yolo_img_dir: str, labeled_label_dir: str, split_ptg: float, method: str, entropy : bool = False):
    """
    Move a subset of uncertain samples to the labeled pool based on uncertainty scores.

    This function selects a subset of images from the unlabeled dataset based on
    uncertainty scores computed by the provided sampler. The corresponding images
    and label files are moved into the labeled dataset directories. The number of
    moved samples is determined by the product of the current training CSV size
    and a user-specified split ratio.

    Parameters
    ----------
    sampler : UncertaintySampler
        The uncertainty sampling interface used for selecting samples
    train_csv : str
        Path to the current training CSV file, used to estimate dataset size
    unlabeled_yolo_img_dir : str
        Directory containing unlabeled images
    unlabeled_label_dir : str 
        Directory containing pseudo-label files for the unlabeled images
    labeled_yolo_img_dir : str 
        Destination directory for labeled images
    labeled_label_dir : str 
        Destination directory for labeled label files
    split_ptg : float
        Fraction of the total training set size that determines how many
        samples are moved from unlabeled to labeled
    method : str
        Pooling method used to compute uncertainty scores (e.g., 'avg', 'min')

    Notes
    -----
    - Files are physically moved using `shutil.move`.
    - Missing files are skipped with a warning.
    - The move count is capped by the number of available unlabeled images.

    Raises
    ------
    FileNotFoundError
        If any expected file is missing during the move operation.
    """

    if entropy:
        scores = sampler.yolo_entropy_pooling(unlabeled_yolo_img_dir, method)
    else:
        scores = sampler.yolo_conf_pooling(unlabeled_yolo_img_dir, method)

    train_csv = Path(train_csv)
    unlabeled_yolo_img_dir = Path(unlabeled_yolo_img_dir)
    unlabeled_label_dir = Path(unlabeled_label_dir)
    labeled_yolo_img_dir = Path(labeled_yolo_img_dir)
    labeled_label_dir = Path(labeled_label_dir)
    total_csv_rows = count_csv_rows(train_csv)
    total_unlabeled_images = len(list(unlabeled_yolo_img_dir.glob('*.jpg')))
    move_count = min(total_unlabeled_images, int(split_ptg * total_csv_rows))
    selected_samples = [Path(s[0]) for s in scores[:move_count]]

    for img_file in selected_samples:
        label_file = img_file.with_suffix('.txt')
        src_img_yolo = unlabeled_yolo_img_dir / img_file.name
        src_lbl = unlabeled_label_dir / label_file.name
        dst_img_yolo = labeled_yolo_img_dir / img_file.name
        dst_lbl = labeled_label_dir / label_file.name

        try:
            shutil.move(src_img_yolo, dst_img_yolo)
            shutil.move(src_lbl, dst_lbl)
            logging.info(f'Moved {img_file.name} and {label_file.name} to labeled image/labels directories.')
        except FileNotFoundError as e:
            logging.warning(f'Skipping missing file: {e}')
            