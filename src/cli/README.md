# CLI (cli)
This directory contains the command line interfaces for constructing the dataset ([dataset_constructor](https://github.com/UCSD-E4E/AL_toolkit/blob/main/src/cli/dataset_constructor_cli.py)), preprocessing the dataset ([yolo_data_setup](https://github.com/UCSD-E4E/AL_toolkit/blob/main/src/cli/yolo_data_setup_cli.py)), and training ([trainer](https://github.com/UCSD-E4E/AL_toolkit/blob/main/src/cli/yolo_trainer_cli.py)).

## Running CLI
```bash
cd path/to/AL_toolkit
python -m src.cli.cli_file --config path/to/config/config_file.yaml
