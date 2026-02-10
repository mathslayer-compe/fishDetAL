import os
from src.models.model_handler import YOLOModelHandler
from typing import List, Tuple
import random

class RandomSampler:
    """
    Implements random sampling strategy for active learning.
    This class uses a YOLOModelHandler instance to perform predictions and
    aggregates detection confidence scores by assigning a uniform score.
    It is used in active learning pipelines to randomly select samples
    for annotation.
    Attributes
    ----------
    yolo_model_handler : YOLOModelHandler
        An instance of YOLOModelHandler used to generate YOLO predictions   
    """
    def __init__(self, yolo_model_handler: YOLOModelHandler):
        self.yolo_model_handler = yolo_model_handler

    def random_sampling(self, source: str) -> List[Tuple[str, float]]:
        """
        Assign uniform confidence scores for predictions from a dataset.
        This method runs the model on the given source and assigns a uniform
        confidence score (0) per image.
        Parameters
        ----------
        source : str
            Path to the image directory or dataset to predict on.
        Returns
        -------
        conf_scores: List[Tuple[str, float]]
            A list of tuples where each tuple contains:
            - The image filename (str)
            - The assigned confidence score (float).
        """
        conf_scores = []

        for result in self.yolo_model_handler.predict(source):
            filename = os.path.basename(result.path)
            conf_scores.append([filename, 0]) # Confidence score is irrelevant for random sampling

        random.seed(42)
        random.shuffle(conf_scores)      
        return conf_scores
    