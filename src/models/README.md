# Models (models)
This directory contains the model handler for the YOLO model as well as the trainer for the model. More changes and additions will be made as new models are added. For now, all YOLO models will be supported.

## Running Training
```bash
cd path/to/AL_toolkit
python -m src.cli.trainer_cli --config path/to/training_config_yolo.yaml
```

## Model Zoo
The following table contains weights for the trained models and the sampling methods they were trained on. The best performance came from Entropy-based Sampling with Maximum Entropy aggregation with the highest mean mAP50-95 out of 5 seeds. The performance however was not statistically significant enough to be considered different from Least Confidence Sampling with Minimum Confidence-Score aggregation as the p-value wasn't less than 0.05 in the paired t-tests, so this model will be included as well with the goal of assessing performance during deployment onto Label Studio or the application to determine a clearer winner out of these methods. 

Download the models and set their path next to the 'model' parameter in the YOLO training YAML config file to run the corresponding Active Learning sampling method on a model with the pretrained weights shown below. 

| Method                       | Model Weights
|-------------------------------|----------------
|Entropy-based Sampling (Maximum Aggregation) | [Ent-Max Best Model Weights Download Link](https://drive.google.com/uc?export=download&id=1GKFHhQCOVvJmLIbuOwvwj-nzzoy9h85w) 
|Least Confidence Sampling (Minimum Aggregation) | [LC-Min Best Model Weights Download Link](https://drive.google.com/uc?export=download&id=1aoYgWLj6YzecFnCItP_STC1BMRj5LpLC)