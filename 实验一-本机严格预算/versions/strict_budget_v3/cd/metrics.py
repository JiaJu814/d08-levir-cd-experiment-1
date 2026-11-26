"""像素混淆矩阵跨整集合累积，边界F1按整幅图像计数后汇总。"""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def ratio(x, y):
    return float(x / y) if y else None


def edge(mask):
    return mask & ~binary_erosion(mask, border_value=1)


class Metrics:
    def __init__(self, tolerance=2):
        self.tolerance = tolerance
        self.tp = self.fp = self.fn = self.tn = 0
        self.pred_edges = self.true_edges = self.match_pred = self.match_true = 0

    def update(self, pred, true):
        pred, true = np.asarray(pred, bool), np.asarray(true, bool)
        self.tp += int((pred & true).sum())
        self.fp += int((pred & ~true).sum())
        self.fn += int((~pred & true).sum())
        self.tn += int((~pred & ~true).sum())
        p, t = edge(pred), edge(true)
        if self.tolerance:
            pd = binary_dilation(p, iterations=self.tolerance)
            td = binary_dilation(t, iterations=self.tolerance)
        else:
            pd, td = p, t
        self.pred_edges += int(p.sum())
        self.true_edges += int(t.sum())
        self.match_pred += int((p & td).sum())
        self.match_true += int((t & pd).sum())

    def result(self):
        bp, br = ratio(self.match_pred, self.pred_edges), ratio(self.match_true, self.true_edges)
        bf = None if not (self.pred_edges or self.true_edges) else (2 * bp * br / (bp + br) if bp is not None and br is not None and bp + br else 0.0)
        return {'precision': ratio(self.tp, self.tp + self.fp), 'recall': ratio(self.tp, self.tp + self.fn),
                'f1': ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
                'iou': ratio(self.tp, self.tp + self.fp + self.fn),
                'false_positive_rate': ratio(self.fp, self.fp + self.tn),
                'false_negative_rate': ratio(self.fn, self.fn + self.tp),
                'boundary_f1': bf, 'boundary_precision': bp, 'boundary_recall': br,
                'tp': self.tp, 'fp': self.fp, 'fn': self.fn, 'tn': self.tn}
