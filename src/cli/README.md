# CLI (cli)
This directory contains the command line interfaces for constructing the dataset ([dataset_constructor](https://github.com/mathslayer-compe/fishDetAL/blob/main/src/cli/dataset_constructor_cli.py)), preprocessing the dataset ([yolo_data_setup](https://github.com/mathslayer-compe/fishDetAL/blob/main/src/cli/yolo_data_setup_cli.py)), and training ([trainer](https://github.com/mathslayer-compe/fishDetAL/blob/main/src/cli/trainer_cli.py)).

## Running CLI
```bash
cd path/to/fishDetAL
python -m src.cli.cli_file --config path/to/config/config_file.yaml

