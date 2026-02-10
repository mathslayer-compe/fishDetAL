from src.dataset_tools.dataset_constructor import DeepFishConstructor
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Setup DeepFish dataset for Active Learning')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir    
    constructor_object = DeepFishConstructor(dataset_dir)
    constructor_object.construct()
    logging.info('Finished loading DeepFish dataset into AL-ready format')

if __name__ == "__main__":
    main()
