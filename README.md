# Active Learning for Underwater Fish Detection (fishDetAL)
## Abstract 
Accurate fish detection in underwater imagery is essential for non-invasive fisheries monitoring and conservation, yet the scarcity of annotated underwater datasets poses a significant challenge for training robust computer vision models. This study evaluates uncertainty-based active learning strategies for improving fish detection performance in data-scarce environments, with application to the FishSense Mobile fish measurement system. Using the DeepFish dataset and a lightweight YOLOv8n detector, five querying strategies are compared—random sampling, least confidence sampling with mean and minimum aggregation, and entropy-based sampling with mean and maximum aggregation—across multiple training iterations under a fixed annotation budget. Results demonstrate that extremal aggregation strategies (minimum confidence and maximum entropy) significantly outperform both random sampling and mean aggregation approaches, reaching mean mAP50-95 of 0.579 and 0.578 compared to 0.564 for random sampling. Despite selecting largely non-overlapping image sets, both extremal methods converged on the same challenging environmental conditions, particularly sparse algal bed habitats, suggesting that habitat-driven visual difficulty dominates uncertainty-based sample selection in underwater detection tasks. These findings provide practical guidance for active learning in underwater fish detection and potentially other single-class, dense object detection scenarios, demonstrating that simple uncertainty sampling with extremal aggregation can improve performance over random and mean-aggregated strategies, though environmental stratification may require additional diversity mechanisms for optimal coverage.

## Creating + Training Model
1) Construct DeepFish a dataset that can be used for YOLO training
    ```bash
    cd path/to/AL_toolkit
    python -m src.cli.dataset_constructor_cli --dataset_dir src/datasets
    ```

2) Preprocess the dataset to get it ready for YOLO training
    ```bash
    cd path/to/AL_toolkit
    python -m src.cli.yolo_data_setup_cli --config path/to/yolo_dataset_config.yaml
    ```

3) Run the trainer
    ```bash
    cd path/to/AL_toolkit
    python -m src.cli.trainer_cli --config path/to/training_config_yolo.yaml 
    ```
    To log training, run the script below:
    ```bash
    cd path/to/AL_toolkit
    python -m src.cli.trainer_cli --config path/to/training_config_yolo.yaml 2>&1 | tee src/models/training.log
    ```

## Predictions + Inference
1) Generate predictions on test images
    ```bash
    yolo detect predict model=path/to/best/model source=path/to/AL_Train/test/images save=True
    ```
2) Generate metrics on test set
    ```bash
    yolo detect val model=path/to/best/model data=path/to/AL_Train/data.yaml split=test 
    ```
## Results
To learn more about the results, visit the [samplers](https://github.com/mathslayer-compe/fishDetAL/tree/main/src/samplers) directory. To download the best model weights for training, visit the [models](https://github.com/mathslayer-compe/fishDetAL/tree/main/src/models) directory.

