import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics._base import type_of_target

class Accuracy:
    """
    Computes how often predictions equals true labels.

    Args:
        y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
        y_pred: Tensor of The predicted values. shape = [batch_size, d0, .., dN]
        threshold: Threshold value for binary or multi-label logits. default: `0.5`
        from_logits: If the predictions are logits/probabilites or actual labels. default: `True`
            * `True` for Logits
            * `False` for Actual labels

    Returns:
        Tensor of Accuracy metric
    """

    def __init__(self, threshold: float = 0.5, from_logits: bool = True):
        self.threshold = threshold
        self.from_logits = from_logits

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            y_pred, y_true = self._conversion(y_pred, y_true, self.threshold)
        return torch.mean((y_pred == y_true).float())

    def _conversion(self, y_pred, y_true, threshold):
        if len(y_pred.shape) == len(y_true.shape) + 1:
            y_pred = torch.argmax(y_pred, dim=1)

        if len(y_pred.shape) == len(y_true.shape) and y_pred.dtype == torch.float:
            y_pred = (y_pred >= threshold).float()

        return y_pred, y_true


class Precision:
    """
    Computes precision of the predictions with respect to the true labels.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`

    Returns:
        Tensor of precision score
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        true_positives = torch.sum(torch.round(torch.clip(y_pred * y_true, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + self.epsilon)
        return precision


class Recall:
    """
    Computes recall of the predictions with respect to the true labels.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`

    Returns:
        Tensor of recall score
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        true_positives = torch.sum(torch.round(torch.clip(y_pred * y_true, 0, 1)))
        actual_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        recall = true_positives / (actual_positives + self.epsilon)
        return recall


class F1Score:
    """
    Computes F1-score between `y_true` and `y_pred`.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`

    Returns:
        Tensor of F1-score
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon
        self.precision = Precision()
        self.recall = Recall()

    def __call__(self, y_pred, y_true):
        precision = self.precision(y_pred, y_true)
        recall = self.recall(y_pred, y_true)
        return 2 * ((precision * recall) / (precision + recall + self.epsilon))
    

class Metric:
    def __init__(self):
        raise NotImplementedError
    
    def add_data(self, pred, real):
        raise NotImplementedError
    
    def get_metric_value(self):
        raise NotImplementedError
    
class MyAccuracy(Metric):
    def __init__(self):
        self.total = 0
        self.correct = 0
    
    def add_data(self, pred, real):
        pred = pred.argmax(1)
        for i in range(len(real)):
            if real[i] == 17:
                continue
            self.total += 1
            if real[i] == pred[i]:
                self.correct += 1
        
    
    def get_metric_value(self):
        return self.correct * 100 / self.total

class MyAccuracyAll(Metric):
    def __init__(self):
        self.total = 0
        self.correct = 0
    
    def add_data(self, pred, real):
        pred = pred.argmax(1)
        for i in range(len(real)):
            self.total += 1
            if real[i] == pred[i]:
                self.correct += 1
        
    def get_metric_value(self):
        return self.correct * 100 / self.total

class MyF1Score(Metric):
    def __init__(self):
        self.pred = []
        self.real = []
    
    def add_data(self, pred, real):
        self.pred.extend(map(int, pred.argmax(1)))
        self.real.extend(map(int, real))
    
    def get_metric_value(self):
        return f1_score(self.real, self.pred, average='weighted', zero_division=0)

class MyPrecission(Metric):
    def __init__(self):
        self.pred = []
        self.real = []
    
    def add_data(self, pred, real):
        self.pred.extend(map(int, pred.argmax(1)))
        self.real.extend(map(int, real))
        
    
    def get_metric_value(self):
        return precision_score(self.real, self.pred, average='weighted', zero_division=0)

class MyRecall(Metric):
    def __init__(self):
        self.pred = []
        self.real = []
    
    def add_data(self, pred, real):
        self.pred.extend(map(int, pred.argmax(1)))
        self.real.extend(map(int, real))
    
    def get_metric_value(self):
        return recall_score(self.real, self.pred, average='weighted', zero_division=0)

