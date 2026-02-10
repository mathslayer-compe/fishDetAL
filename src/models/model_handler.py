import logging
from pathlib import Path
from ultralytics import YOLO
import shutil

class YOLOModelHandler:
    """
    A handler class for managing YOLO model training, evaluation, prediction, and checkpoint operations.

    This class provides a simple interface to interact with the Ultralytics YOLO model, including:
    - Training a model on labeled data
    - Evaluating its performance
    - Running predictions
    - Saving and reloading the best-performing weights

    Attributes
    ----------
    model_path : Path
        Path to the YOLO model file (.pt).
    model : YOLO
        The YOLO model instance loaded from the specified path.
    """
    def __init__(self, model_path: str):
        """
        Initialize the ModelHandler with a YOLO model.

        Parameters
        ----------
        model_path : str 
            Path to the pretrained YOLO model file.
        """
        self.model_path = Path(model_path) 
        self.model = YOLO(self.model_path)

    def train(self, data: str, epochs: int, project: str, name: str):
        """
        Train the YOLO model on a dataset.

        Parameters
        ----------
        data : str
            Path to the dataset YAML file.
        epochs : int
            Number of training epochs.
        project : str
            Directory where training results will be stored.
        name : str
            Subdirectory name for this particular training run.

        Returns
        -------
        results
            The training results object returned by YOLO.
        """
        logging.info(f'Starting training')
        results = self.model.train(data=data, epochs=epochs, project=project, name=name, exist_ok=True)
        logging.info(f'Training completed. Results saved under {project}/{name}')
        return results
    

    def evaluate(self, data: str):
        """
        Evaluate the trained YOLO model on a validation dataset.

        Parameters
        ----------
        data : str
            Path to the dataset YAML file used for evaluation.

        Returns
        -------
        metrics
            The evaluation metrics object containing mAP and other performance statistics.
        """
        logging.info(f'Starting evaluation')
        metrics = self.model.val(data=data)
        logging.info(f'Evaluation completed. mAP: {metrics.box.map: .4f}')
        return metrics

    def predict(self, source: str, conf: float = 0.02) -> list:
        """
        Run inference using the YOLO model.

        Parameters
        ----------
        source : str
            Path to an image, video, or directory for inference.
        conf : float, optional
            Confidence threshold for predictions (default is 0.02).

        Returns
        -------
        preds: list
            A list of prediction results for the provided source(s).
        """
        return self.model.predict(source=source, conf=conf, stream=True)

    def reload_best(self, project, name):
        """
        Reload the best (or tagged) YOLO model checkpoint.

        Parameters
        ----------
        project : str
            Directory where training results are stored.
        name : str
            Subdirectory name for this particular training run.
        Returns
        -------
        self.model: YOLO
            The reloaded YOLO model instance.
       """
        best_path = Path(project) / name/ 'weights/best.pt'
        logging.info(f'Reloading best model from {best_path}')
        self.model = YOLO(best_path)
        return self.model
    