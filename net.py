import torch.nn as nn


class VBN(nn.Module):
    def __init__(self, inchannel=1) -> None:
        super().__init__()

        # 图像输入 48*48*3
        self.phrase1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(p=0.25),
        )

        self.phrase2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(p=0.25),
        )

        self.phrase3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(p=0.5),
        )

        self.phrase4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )

        self.classifer = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.phrase1(x)
        x = self.phrase2(x)
        x = self.phrase3(x)


        # 后面要接全连接的
        x = x.view(x.size(0),-1)
        # x = x.view(-1, 256)
        x = self.phrase4(x)

        x = self.classifer(x)

        return x
