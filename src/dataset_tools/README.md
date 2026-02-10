# Dataset Tools (dataset_tools)
The purpose of this directory is to perform operations on items in the dataset to make them easier to use for training and 
inference.

## Tools/Operations
The 3 main operations are (follow in order): 
1) [Constructing the DeepFish dataset with a 70-20-10 Train-Val-Test split](https://github.com/UCSD-E4E/AL_toolkit/blob/main/src/dataset_tools/dataset_constructor.py)

2) [Preprocessing dataset to put it into a format that works for YOLO](https://github.com/UCSD-E4E/AL_toolkit/blob/main/src/dataset_tools/data_setup.py) (mainly splitting dataset): 

3) [Moving data from unlabeled to labeled directory during the active learning process](https://github.com/UCSD-E4E/AL_toolkit/blob/main/src/dataset_tools/move_data.py) (simulation of labeling process): 
    

## Running the Operations
1) To construct the DeepFish dataset with a 70-20-10 Train-Val-Test split, run the following commands:
    ```bash
    cd path/to/AL_toolkit
    python -m src.cli.dataset_constructor_cli --dataset_dir path/to/dataset_dir
    ```


2) To preprocess the dataset into a trainable split, run the following commands:
    ```bash
    cd path/to/AL_toolkit
    python -m src.cli.yolo_data_setup_cli --config path/to/yolo_dataset_config.yaml
    ```

3) [move_data](https://github.com/UCSD-E4E/AL_toolkit/blob/main/src/dataset_tools/move_data.py) is run within the training loop itself and will be run during the training
