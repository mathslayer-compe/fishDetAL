from src.models.trainer import YOLOTrainer
import argparse
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Command-line interface for running Active Learning training.

    This function parses a YAML configuration file provided via the CLI,
    initializes the Trainer class with the configuration, and runs the 
    active learning training loop. Progress and completion are logged.

    Command-line Arguments
    ----------------------
    --config : str
        Path to the YAML configuration file containing training parameters.

    Raises
    ------
    FileNotFoundError
        If the specified config file does not exist.
    """
    parser = argparse.ArgumentParser(description='Active Learning Trainer CLI')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml yolo config file')
    args = parser.parse_args()
    config_path_yolo = Path(args.config)

    if not config_path_yolo.exists():
        raise FileNotFoundError(f'Config file not found: {config_path_yolo}')

    with open(config_path_yolo, 'r') as f:
        config_yolo = yaml.safe_load(f)
    
    trainer = YOLOTrainer(config_yolo)    
    trainer.run_AL()
    logging.info("Active Learning training completed successfully.")

if __name__ == "__main__":
    main()
