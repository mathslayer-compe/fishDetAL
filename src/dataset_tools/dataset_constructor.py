import kagglehub
import shutil
import os
from pathlib import Path
import random
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepFishConstructor:
    """
    Handles downloading and preprocessing the DeepFish dataset from Kaggle into a format suitable for YOLO preprocessing.
    
    Attributes
    ----------
    dataset_dir : str
        Directory where the DeepFish dataset will be downloaded and preprocessed.
    Methods
    -------
    download_DeepFish()
        Downloads the DeepFish dataset from Kaggle.
    make_preprocess_dirs()
        Creates necessary directories for preprocessing.
    move_data()
        Moves and organizes data into the created directories.
    split_data(data, train_ratio=0.7, val_ratio=0.2)
        Splits data into training, validation, and test sets.
    get_labels(split, empty_img_dir_list)
        Assigns labels to the data based on whether images are empty or contain fish.
    load_df(split, empty_img_dir_list)
        Loads data into a pandas DataFrame.
    construct()
        Orchestrates the entire preprocessing workflow.
    """
    def __init__(self, dataset_dir):
        """
        Initialize the DeepFishConstructor with the dataset directory.
        """
        self.dataset_dir = dataset_dir

    def download_DeepFish(self):
        """
        Downloads the DeepFish dataset from Kaggle.
        
        Returns
        -------
        str
            Path to the downloaded DeepFish dataset.
        """
        os.makedirs(self.dataset_dir, exist_ok=True)
        path = kagglehub.dataset_download("vencerlanz09/deep-fish-object-detection")
        shutil.move(path, self.dataset_dir)
        logging.info('Downloaded DeepFish dataset from Kaggle')
        return os.path.join(self.dataset_dir, '1/Deepfish')
    
    def make_preprocess_dirs(self):
        """ 
        Creates necessary directories for preprocessing the DeepFish dataset.
        """
        logging.info('Creating Directories for AL Format')
        os.makedirs(os.path.join(self.dataset_dir, 'DeepFish/images/valid'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, 'DeepFish/images/empty'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, 'DeepFish/labels/valid'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, 'DeepFish/labels/empty'), exist_ok=True)
        logging.info('Finished')
    
    def move_data(self) -> tuple:
        """
        Moves and organizes data into the created directories.
        
        Returns
        -------
        train : list
            List of training image filenames.
        val : list
            List of validation image filenames.
        test : list
            List of test image filenames.
        """
        logging.info('Moving files into directory')
        train = []
        test = []
        val = []
        dir = self.download_DeepFish()
        dir_list = os.listdir(dir)
        dst_dir_valid_img = os.path.join(self.dataset_dir, 'DeepFish/images/valid')
        dst_dir_empty_img = os.path.join(self.dataset_dir, 'DeepFish/images/empty')
        dst_dir_valid_lbl = os.path.join(self.dataset_dir, 'DeepFish/labels/valid')
        dst_dir_empty_lbl = os.path.join(self.dataset_dir , 'DeepFish/labels/empty')

        for i in range(len(dir_list)-1):
            if i != len(dir_list)-2:
                total_file_list_train = os.listdir(os.path.join(dir, dir_list[i], 'train')) 
                total_file_list_valid = os.listdir(os.path.join(dir, dir_list[i], 'valid'))
                total_file_list = total_file_list_train + total_file_list_valid
                stem_list = [f for f in total_file_list if Path(f).suffix == '.jpg']
                file_train, file_val, file_test = self.split_data(stem_list)
                train.extend(file_train)
                test.extend(file_test)
                val.extend(file_val)

                for file in total_file_list:
                    filepath = Path(file)
            
                    if filepath.suffix == '.jpg':
                        if file in total_file_list_train:
                            shutil.copy(os.path.join(dir, dir_list[i], 'train', file), dst_dir_valid_img)
                        else:
                            shutil.copy(os.path.join(dir, dir_list[i], 'valid', file), dst_dir_valid_img)
                    else:
                        if file in total_file_list_train:
                            shutil.copy(os.path.join(dir, dir_list[i], 'train', file), dst_dir_valid_lbl)
                        else:
                            shutil.copy(os.path.join(dir, dir_list[i], 'valid', file), dst_dir_valid_lbl)
            else:
                for directory in os.listdir(os.path.join(dir, dir_list[i])):
                    total_file_list = os.listdir(os.path.join(dir, dir_list[i], directory))
                    stem_list = [f for f in total_file_list if Path(f).suffix == '.jpg']
                    file_train, file_val, file_test = self.split_data(stem_list)
                    train.extend(file_train)
                    test.extend(file_test)
                    val.extend(file_val)

                    for file in total_file_list:
                        filepath = Path(file)

                        if filepath.suffix == '.jpg':
                            shutil.copy(os.path.join(dir, dir_list[i], directory, file), dst_dir_empty_img)
                        else:
                            shutil.copy(os.path.join(dir, dir_list[i], directory, file), dst_dir_empty_lbl)

        logging.info('Finished')
        return train, val, test

    def split_data(self, data, train_ratio=0.7, val_ratio=0.2):
        n = len(data)
        random.seed(42) #1:42, 2:21, 3:69, 4:10, 5:99
        random.shuffle(data)
        n_train = int(n*train_ratio)
        n_val = int(n*val_ratio)
        train = data[:n_train]
        val = data[n_train:n_train+n_val]
        test = data[n_train+n_val:]
        return train, val, test
    
    def get_labels(self, split, empty_img_dir_list):
        new_split = []

        for i in range(len(split)):
            if split[i] in empty_img_dir_list:
                new_split.append([0, split[i]])
            else:
                new_split.append([1, split[i]])
        
        return new_split
    
    def load_df(self, split, empty_img_dir_list):
        df = pd.DataFrame(columns=['ID', 'labels', 'classes', 'frames'])
        new_split = self.get_labels(split, empty_img_dir_list)

        for i in range(len(new_split)):
            labels = new_split[i][0]
            frames = Path(new_split[i][1]).stem
            
            if new_split[i][0] == 0:
                classes = 'empty'
                id = f'empty/{Path(new_split[i][1]).stem}'
            else:
                classes = 'valid'
                id = f'valid/{Path(new_split[i][1]).stem}'
        
            new_row = {'ID':id, 'labels':labels, 'classes':classes, 'frames':frames}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
        return df
    
    def construct(self):
        self.make_preprocess_dirs()
        train, val, test = self.move_data()
        empty_img_dir_list = os.listdir(os.path.join(self.dataset_dir, 'DeepFish/images/empty'))
        valid_img_dir_list = os.listdir(os.path.join(self.dataset_dir, 'DeepFish/images/valid'))
        data = empty_img_dir_list + valid_img_dir_list
        logging.info('Splitting data into train-test-val')
        # train, test, val = self.split_data(data)
        logging.info('Done')
        logging.info('Creating CSV Files')
        df_train = self.load_df(train, empty_img_dir_list)
        df_test = self.load_df(test, empty_img_dir_list)
        df_val = self.load_df(val, empty_img_dir_list)
        df_train.to_csv(os.path.join(self.dataset_dir, 'DeepFish/train.csv'))
        df_val.to_csv(os.path.join(self.dataset_dir, 'DeepFish/val.csv'))
        df_test.to_csv(os.path.join(self.dataset_dir, 'DeepFish/test.csv'))
        logging.info('Done')
