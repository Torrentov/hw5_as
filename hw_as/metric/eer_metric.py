import torch
from torch import Tensor

from hw_as.metric.utils import compute_eer
from hw_as.base.base_metric import BaseMetric


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, all_probs, all_targets, **kwargs) -> (float, float):
        eer, thr = compute_eer(
            bonafide_scores=all_probs[all_targets == 1],
            other_scores=all_probs[all_targets == 0]
        )  # нашел в чате как пример правильного использования
        return eer, thr
