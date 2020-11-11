from collections import Counter
from contextlib import suppress

def ZeroDivisionDecorator(func):
    """property to avoid ZeroDivisionError

    Args:
        func (func): a function to wrap with the property
    """
    def func_wrapper(*args, **kwargs):

        with suppress(ZeroDivisionError):
           return func(*args, **kwargs)
    return func_wrapper

class Metrics():
    """metrics calculation for recall, precision and f1 score
    """    
    def __init__(self, y_true: np.array, y_pred: np.array) -> None:
        counts = Counter(zip(y_pred, y_true))
        self.tp = counts[1, 1]
        self.fn = counts[1, 0]
        self.fp = counts[0, 1]

    @property
    @ZeroDivisionDecorator
    def recall(self) -> float:
        """calculate the recall for binary classification

        Returns:
            float: recall score
        """
        return self.tp / (self.tp + self.fn)

    @property
    @ZeroDivisionDecorator
    def precision(self) -> float:
        """calculate the precision for binary classification

        Returns:
            float: precision score
        """    
        return self.tp / (self.tp + self.fp)

    @property
    @ZeroDivisionDecorator
    def f1(self) -> float:
        """calculate the f1 score for binary classification

        Returns:
            float: f1 score
        """    
        p = self.precision()
        r = self.recall()
        if p and r:
            return 2 * ((p * r) / (p + r))