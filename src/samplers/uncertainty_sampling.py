import os
import logging
from src.models.model_handler import YOLOModelHandler
import numpy as np
from typing import List, Tuple
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image

class UncertaintySampler:
    """
    A class for computing uncertainty-based confidence scores from model predictions

    This class uses a ModelHandler instance to perform predictions and
    aggregates detection confidence scores using various pooling strategies.
    It is used in active learning pipelines to select uncertain samples
    for annotation.

    Attributes
    ----------
    yolo_model_handler : ModelHandler
        An instance of ModelHandler used to generate YOLO predictions
            An instance of ModelHandler used to generate EfficientNet predictions
    """
    def __init__(self, yolo_model_handler: YOLOModelHandler):
        """
        Initialize the UncertaintySampler with a model handler.

        Parameters
        ----------
        yolo_model_handler : ModelHandler
            The ModelHandler object that provides access to YOLO predictions.
        """
        self.yolo_model_handler = yolo_model_handler 
    
    def yolo_conf_pooling(self, source, method: str) -> List[Tuple[str, float]]:
        """
        Compute aggregated confidence scores for predictions from a dataset.

        This method runs the model on the given source and computes a single
        confidence score per image using one of several aggregation methods.

        Parameters
        ----------
        source : str
            Path to the image directory or dataset to predict on.
        method : str
            Aggregation strategy for combining individual detection confidences.
            Must be one of ['avg', 'min'].

        Returns
        -------
        conf_scores: List[Tuple[str, float]]
            A list of tuples where each tuple contains:
            - The image filename (str)
            - The computed confidence score (float)

        Raises
        ------
        ValueError
            If `method` is not one of the supported pooling strategies.
        """
        valid_methods = ['avg', 'min']
        method = method.lower().strip()

        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        conf_scores = []

        for result in self.yolo_model_handler.predict(source):
            filename = os.path.basename(result.path)
            
            if len(result) == 0:
                conf_scores.append([filename, 1])
                continue

            conf = result.boxes.conf.cpu().numpy()

            if method == 'avg':
                score = np.mean(conf).item()
            elif method == 'min':
                score = np.min(conf).item()
            
            conf_scores.append([filename, score])
        
        conf_scores.sort(key=lambda x: x[1])
        return conf_scores

    def yolo_entropy_pooling(self, source, method: str) -> List[Tuple[str, float]]:
        """
        Compute aggregated entropy scores for predictions from a dataset.

        This method runs the model on the given source and computes a single
        entropy score per image using one of several aggregation methods from an array of confidence scores.

        Parameters
        ----------
        source : str
            Path to the image directory or dataset to predict on.
        method : str
            Aggregation strategy for combining individual detection confidences.
            Must be one of ['avg', 'max'].

        Returns
        -------
        conf_scores: List[Tuple[str, float]]
            A list of tuples where each tuple contains:
            - The image filename (str)
            - The computed confidence score (float)

        Raises
        ------
        ValueError
            If `method` is not one of the supported pooling strategies.
        """
        valid_methods = ['avg', 'max']
        method = method.lower().strip()

        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        results = self.yolo_model_handler.predict(source)
        entropy_scores = []

        for result in results:
            filename = os.path.basename(result.path)

            if len(result) == 0:
                entropy_scores.append([filename, 0])
                continue

            conf = result.boxes.conf.cpu().numpy()
            object_entropy = -((conf * np.log(np.clip(conf, 1e-12, 1.0))) + ((1-conf)*np.log(np.clip(1-conf, 1e-12, 1.0))))

            if method == 'avg':
                score = np.mean(object_entropy).item()
            elif method == 'max':
                score = np.max(object_entropy).item()
            
            entropy_scores.append([filename, score])
        
        entropy_scores.sort(key=lambda x: x[1], reverse=True)
        return entropy_scores
