import os
from src.dataset_tools.move_data import move_data_random, move_data_uncertainty
import logging
from src.models.model_handler import YOLOModelHandler
from src.samplers.uncertainty_sampling import UncertaintySampler
from src.samplers.random_sampling import RandomSampler
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
writer = SummaryWriter('runs/plots')

class YOLOTrainer:
    """
    Handles the active learning training loop for an object detection model.

    This class manages the iterative process of training, evaluating, and 
    selectively moving unlabeled data to the labeled set based on model 
    confidence. The goal is to improve model performance until a target 
    mAP (mean Average Precision) is achieved or improvement stalls.

    Attributes
    ----------
    TARGET_MAP : float
        Target mean Average Precision threshold to stop training.
    IMPROVEMENT_THRESHOLD : float
        Minimum improvement in mAP required to continue training.
    model : str
        Model name or identifier for the experiment.
    data : str
        Path to dataset or dataset configuration.
    epochs : int
        Number of training epochs per iteration (default: 50).
    project : str
        Project directory name used by ModelHandler.
    name : str
        Model name or run identifier used by ModelHandler.
    train_csv : str
        Path to the CSV file listing labeled training samples.
    unlabeled_img_dir : str
        Directory containing unlabeled images.
    unlabeled_label_dir : str
        Directory containing unlabeled labels.
    labeled_img_dir : str
        Directory containing labeled images.
    labeled_label_dir : str
        Directory containing labeled labels.
    yolo_model_handler : ModelHandler
        Wrapper around the model to handle training, evaluation, and saving.
    entropy : bool
        Flag indicating whether to use entropy-based sampling.
    method : str
        Sampling method to use for selecting samples.
    random: bool
        Flag indicating whether to use random sampling.
    """

    def __init__(self, config: dict):
        """
        Initialize the Trainer with a configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing all required parameters for training:
            {
                'model': str,
                'data': str,
                'epochs': int,
                'project': str,
                'name': str,
                'train_csv': str,
                'unlabeled_img_dir': str,
                'unlabeled_label_dir': str,
                'labeled_img_dir': str,
                'labeled_label_dir': str,
                'method': str,
                'entropy': bool,
                'random': bool
            }
        """
        self.model = config['model']
        self.data = config['data']
        self.epochs = config.get('epochs', 50)
        self.project = config['project']
        self.name = config['name']
        self.train_csv = config['train_csv']
        self.unlabeled_img_dir = config['unlabeled_img_dir']
        self.unlabeled_label_dir = config['unlabeled_label_dir']
        self.labeled_img_dir = config['labeled_img_dir']
        self.labeled_label_dir = config['labeled_label_dir']
        self.method = config['method']
        self.entropy = bool(config.get('entropy', False))
        self.random = bool(config.get('random', False))
        self.yolo_model_handler = YOLOModelHandler(self.model)
        self.random_sampler = RandomSampler(self.yolo_model_handler)
        self.uncertainty_sampler = UncertaintySampler(self.yolo_model_handler)

    def run_AL(self):
        """
        Execute the full Active Learning training loop.

        Trains the model iteratively, evaluates its performance, and moves 
        the least confident samples from the unlabeled to the labeled set 
        after each iteration until performance converges or the target mAP 
        is reached.

        Workflow:
        ----------
        1. Train model on current labeled data.
        2. Evaluate model and record mAP.
        3. If mAP < TARGET_MAP:
            - Compute split percentage.
            - Move uncertain unlabeled samples into labeled set.
            - Retrain and re-evaluate.
        4. Stop if improvement is insufficient or target mAP is achieved.
        """
        if self.random and self.entropy:
            raise KeyError('Random and Entropy can not be true at the same time. Only one can be true')
        
        logging.info('Starting Initial Training Cycle')
        iteration = 1
        self.yolo_model_handler.train(self.data, self.epochs, self.project, self.name)
        metrics = self.yolo_model_handler.evaluate(self.data)
        curr_map = metrics.box.map.item()
        writer.add_scalar('mAP50-95', curr_map, iteration-1)
        f1 = metrics.box.f1.mean().item()
        writer.add_scalar('F1', f1, iteration-1)
        precision = metrics.box.mp.item()
        writer.add_scalar('Precision', precision, iteration-1)
        recall = metrics.box.mr.item()
        writer.add_scalar('Recall', recall, iteration-1)
        map50 = metrics.box.map50.item()
        writer.add_scalar('mAP50', map50, iteration-1)
        
        logging.info(f'Initial Training Cycle Completed with mAP: {curr_map:.4f} and F1: {f1:.4f}')
        self.yolo_model_handler.reload_best(self.project, self.name)

        for i in range(3):
            split_ptg = 0.05
            logging.info(f"Iteration {iteration}: mAP={curr_map:.4f}, split={split_ptg*100:.1f}%")
            iteration += 1

            if self.random:
                move_data_random(self.random_sampler, self.train_csv, self.unlabeled_img_dir, self.unlabeled_label_dir, self.labeled_img_dir, self.labeled_label_dir, split_ptg)
            else:
                move_data_uncertainty(self.uncertainty_sampler, self.train_csv, self.unlabeled_img_dir, self.unlabeled_label_dir, self.labeled_img_dir, self.labeled_label_dir, split_ptg, self.method, self.entropy)

            self.yolo_model_handler.train(self.data, self.epochs, self.project, self.name) 
            metrics = self.yolo_model_handler.evaluate(self.data)
            curr_map = metrics.box.map.item()
            writer.add_scalar('mAP50-95', curr_map, iteration-1)
            f1 = metrics.box.f1.mean().item()
            writer.add_scalar('F1', f1, iteration-1)
            precision = metrics.box.mp.item()
            writer.add_scalar('Precision', precision, iteration-1)
            recall = metrics.box.mr.item()
            writer.add_scalar('Recall', recall, iteration-1)
            map50 = metrics.box.map50.item()
            writer.add_scalar('mAP50', map50, iteration-1)
            logging.info(f'Model saved with mAP: {curr_map:.4f} and F1: {f1:.4f}')
            self.yolo_model_handler.reload_best(self.project, self.name)
    
        logging.info(f'Active Learning completed. Best mAP achieved: {curr_map:.4f}. F1: {f1:.4f}')
