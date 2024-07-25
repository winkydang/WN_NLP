## 以下代码直接使⽤即可
import torch
from torch import nn


class LabelSmoothingCELoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCELoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # .scatter_也是⼀种数据填充⽅法，⽬的仍然是将self.confidence填充到true_dist中
            # 第⼀个参数0/1代表填充的轴，⼤多数情况下使⽤scatter_都使⽤纵轴（1）填充
            # 第⼆个参数就是self.confidence的填充规则，即填充到第⼏列⾥⾯，如[[1], [2]]代表填充到第⼆列和第三列⾥⾯
            # 第三个参数就是填充的数值，int/float
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == "__main__":
    predict = torch.FloatTensor([[1, 1, 1, 1, 1]])
    target = torch.LongTensor([2])
    LSL = LabelSmoothingCELoss(3, 0.03)  # 标签平滑化
    print(LSL(predict, target))  # tensor(1.6577)

