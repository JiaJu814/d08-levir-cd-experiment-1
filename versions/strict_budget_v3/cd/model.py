"""保留作者 FC-Siam-diff 主体，用 log p1 - log p0 得到二分类 logit。"""
from .official_siamunet_diff import SiamUnet_diff

class ChangeDetector(SiamUnet_diff):
    def __init__(self):
        super().__init__(3, 2)

    def forward(self, a, b):
        log_prob = super().forward(a, b)
        return log_prob[:, 1:2] - log_prob[:, 0:1]
