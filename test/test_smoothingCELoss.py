# import torch
#
# # 初始化一个全0的张量，这代表了我们将要填充的新分布。
# # 它的形状是(batch_size, num_classes)。
# true_dist = torch.zeros((3, 5))
#
# # 我们的目标标签，每个样本一个标签。
# target = torch.tensor([2, 1, 4])
#
# # 标签平滑后的信心度
# confidence = 0.8
#
# # 将true_dist所有位置填充标签平滑的值。
# true_dist.fill_(0.05)
#
# # 将 置信度 confidence 填充到 true_dist 的目标位置
# # 使用scatter_来更新我们的true_dist张量。
# # 第一个参数是维度，我们想要沿着类别的维度更新，所以是1。
# # 第二个参数是一个包含索引的张量，指示了要更新的位置。我们需要在第二个维度上添加索引。
# # 第三个参数是我们希望在这些位置上填充的值。
# true_dist.scatter_(1, target.unsqueeze(1), confidence)
#
# # # 确保我们在正确的位置保持原来的信心度。
# # for i in range(target.size(0)):
# #     true_dist[i, target[i]] = confidence
#
# print(true_dist)
import torch
from torch import nn

## 以下代码直接使⽤即可
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
            # 第⼆个参数就是self.confidence的填充规则，即填充到第⼏列⾥⾯，如[[1],[2]]代表填充到第⼆列和第三列⾥⾯
            # 第三个参数就是填充的数值，int/float
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == "__main__":
    predict = torch.FloatTensor([[1, 1, 1, 1, 1]])
    target = torch.LongTensor([2])
    LSL = LabelSmoothingCELoss(3, 0.03)
    print(LSL(predict, target))

