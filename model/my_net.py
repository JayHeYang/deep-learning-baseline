
import torch.nn as nn
from torchvision import models as M


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        resnet18 = M.resnet18(pretrained=False)
        # 更改ResNet18最后全连接层的节点数以适应cifar10
        resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
        self.resnet = resnet18

    def forward(self, x):
        return self.resnet(x)


# if __name__ == '__main__':
#     # Example
#     net = ResNet18(10)
#     print(net)

