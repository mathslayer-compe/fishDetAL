from pathlib import Path
import os
import cv2
import pandas as pd
import logging
from typing import List

class YOLODataSetup:
    """
    Handles setup of a YOLO dataset for Active Learning. This includes:
    - Creating necessary directory structure for labeled, unlabeled, validation, and test sets
    - Splitting the dataset into labeled/unlabeled based on a configurable fraction
    - Copying images and labels to appropriate directories
    - Generating a YOLO-compatible data YAML file

    Attributes
    ----------
    img_path : Path
        Base path to source images
    label_path : Path
        Base path to source labels
    train_csv : Path
        CSV file path containing training dataset information
    val_csv : Path
        CSV file path containing validation dataset information
    test_csv : Path
        CSV file path containing test dataset information
    split_ptg : float
        Fraction of the training dataset to use as labeled data
    num_classes : int
        Number of object classes
    class_names : List[str]
        Names of object classes
    output_dir : Path
        Root directory for all Active Learning dataset splits
    subdirs : List[str]
        Subdirectories to create under `output_dir`
    """
    def __init__(self, config: dict):
        """
        Initialize the YOLO dataset setup with configuration parameters

        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            - image_path: path to source images
            - label_path: path to source labels
            - train_csv, val_csv, test_csv: CSV paths
            - num_classes: number of classes
            - class_names: list of class names
            - split_ptg: fraction of training data to label (optional, default 0.1)
        """
        self.img_path = Path(config['image_path'])
        self.label_path = Path(config['label_path'])
        self.train_csv = Path(config['train_csv'])
        self.val_csv = Path(config['val_csv'])
        self.test_csv = Path(config['test_csv'])
        self.split_ptg = config.get('split_ptg', 0.05)  # default to 5% if not specified
        self.num_classes = config['num_classes']
        self.class_names = config['class_names']
        self.output_dir = self.img_path.parent.parent / 'AL_Train'
        self.subdirs = [
            "val/images",
            "val/labels",
            "test/images",
            "test/labels",
            "labeled/images",
            "labeled/labels",
            "unlabeled/images",
            "unlabeled/labels",
        ]
    
    def setup_dirs(self):
        """
        Create all necessary subdirectories under `output_dir` if they do not exist
        """
        for subdir in self.subdirs:
            os.makedirs(self.output_dir / subdir, exist_ok=True)

    def write_image_and_label(self, src_img_path: Path, src_label_path: Path, dest_img_path: Path, dest_label_path: Path):
        """
        Copy an image and its corresponding label file to the destination paths

        Parameters
        ----------
        src_img_path : Path
            Source image path
        src_label_path : Path
            Source label path
        dest_img_path : Path
            Destination image path
        dest_label_path : Path
            Destination label path

        Raises
        ------
        FileNotFoundError
            If the source image cannot be read
        """
        img = cv2.imread(str(src_img_path))

        if img is None:
            raise FileNotFoundError(f"Image file {src_img_path} not found or could not be opened.")
        
        cv2.imwrite(str(dest_img_path), img)

        with open(src_label_path, 'r') as src_file, open(dest_label_path, 'w') as dest_file:
            dest_file.write(src_file.read())
    
    def process_split(self, df: pd.DataFrame, img_dest_dir: Path, label_dest_dir: Path):
        """
        Process a DataFrame of images and labels, copying them to the specified directories

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'frames' and 'classes' columns
        img_dest_dir : Path
            Destination directory for images
        label_dest_dir : Path
            Destination directory for labels
        """
        extension = '.jpg'
        for frame, cls in zip(df['frames'], df['classes']):
            src_img_path = None
            for ext in [".jpg", ".png"]:
                candidate = self.img_path / cls / f"{frame}{ext}"
                if candidate.exists():
                    src_img_path = candidate
                    extension = ext
                    break

            if src_img_path is None:
                raise FileNotFoundError(f"No image found for {frame} in class {cls}")

            src_label_path = self.label_path / cls / f"{frame}.txt"
            dest_img_path = img_dest_dir / f"{frame}{extension}"   
            dest_label_path = label_dest_dir / f"{frame}.txt"
            logging.info(f'Moved {frame} to YOLO directory')

            self.write_image_and_label(src_img_path, src_label_path, dest_img_path, dest_label_path)
    
    def copy_files(self):
        """
        Split the training dataset into labeled/unlabeled subsets, shuffle it,
        and copy files for train, validation, and test sets
        """
        df_train = pd.read_csv(self.train_csv)
        df_val = pd.read_csv(self.val_csv)
        df_test = pd.read_csv(self.test_csv)

        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)  #1:42, 2:21, 3:69, 4:10, 5:99
        split_idx = int(self.split_ptg * len(df_train))
        df_labeled = df_train.iloc[:split_idx].reset_index(drop=True)
        df_unlabeled = df_train.iloc[split_idx:].reset_index(drop=True)

        self.process_split(df_labeled, self.output_dir / 'labeled/images', self.output_dir / 'labeled/labels')
        self.process_split(df_unlabeled, self.output_dir / 'unlabeled/images', self.output_dir / 'unlabeled/labels')
        self.process_split(df_val, self.output_dir / 'val/images', self.output_dir / 'val/labels')
        self.process_split(df_test, self.output_dir / 'test/images', self.output_dir / 'test/labels')

    def create_yaml(self, num_classes: int, class_names: List[str]):
        """
        Create a YOLO data YAML file specifying train, val, test directories,
        number of classes, and class names

        Parameters
        ----------
        num_classes : int
            Number of object classes
        class_names : List[str]
            Names of the object classes
        """
        yaml_path = self.output_dir / 'data.yaml'

        with open(yaml_path, 'w') as f:
            f.write(f'train: labeled/images\n')
            f.write(f'val: val/images\n')
            f.write(f'test: test/images\n')
            f.write(f'nc: {num_classes}\n')
            f.write(f'names: {class_names}\n')

    def run(self):
        """
        Run the full dataset setup process:
        1. Create directory structure
        2. Copy images and labels to corresponding splits
        3. Generate YOLO-compatible data YAML file
        """
        self.setup_dirs()
        self.copy_files()
        self.create_yaml(self.num_classes, self.class_names)
