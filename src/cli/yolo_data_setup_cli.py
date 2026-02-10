from src.dataset_tools.data_setup import YOLODataSetup
import argparse
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Command-line interface for setting up a YOLO dataset for Active Learning.

    This function:
    1. Parses a YAML config file path from CLI arguments.
    2. Loads the configuration.
    3. Initializes YOLODataSetup with the config.
    4. Runs the setup process to organize images/labels, split train/val/test sets, 
       and create a data.yaml file compatible with YOLO training.
    
    Raises:
        FileNotFoundError: If the provided config file path does not exist.
    """
    parser = argparse.ArgumentParser(description='Setup YOLO dataset for Active Learning')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config file')
    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    setup = YOLODataSetup(config)    
    setup.run()
    logging.info("YOLO dataset setup completed successfully.")

if __name__ == "__main__":
    main()
