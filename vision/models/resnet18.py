from typing import Any, Callable

from .base import DnnModelBase

from torch import nn, device, Tensor


class ResNet18(DnnModelBase):
    """ResNet18モデル"""

    class BasicBlock(nn.Module):
        """ResNet18の残差ブロック"""

        def __init__(
            self, in_channels: int, out_channels: int, stride: int = 1
        ) -> None:
            """コンストラクタ

            Args:
                in_channels (int): 入力チャネル数
                out_channels (int): 出力チャネル数
                stride (int, optional): ストライド
            """
            super().__init__()

            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            self.downsample = None
            if stride > 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x: Tensor) -> Any:
            """順伝搬関数

            Args:
                x (Tensor): 入力データ

            Returns:
                Any: 出力データ
            """
            # 残差接続
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            # ストライドが1以外の場合
            if self.downsample is not None:
                x = self.downsample(x)
            # スキップ接続
            out += x
            out = self.relu(out)
            return out

    def __init__(self, num_classes: int) -> None:
        """コンストラクタ

        Args:
            num_classes (int): クラス数
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            self.BasicBlock(64, 64, 1),
            self.BasicBlock(64, 64, 1),
        )
        self.layer2 = nn.Sequential(
            self.BasicBlock(64, 128, 2),
            self.BasicBlock(128, 128, 1),
        )
        self.layer3 = nn.Sequential(
            self.BasicBlock(128, 256, 2),
            self.BasicBlock(256, 256, 1),
        )
        self.layer4 = nn.Sequential(
            self.BasicBlock(256, 512, 2),
            self.BasicBlock(512, 512, 1),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x: Tensor, return_embed: bool = False) -> Any:
        """順伝搬関数

        Args:
            x (Tensor): 入力データ
            return_embed (bool, optional): 真なら特徴量を、偽ならロジットを返す

        Returns:
            Any: 特徴量またはロジット
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.flatten(1)

        if return_embed:
            return x

        x = self.linear(x)
        return x

    def get_loss_function(self) -> Callable:
        """誤差関数を返す

        Returns:
            Callable: 誤差関数
        """
        return nn.functional.cross_entropy

    def get_device(self) -> device:
        """重みを保持しているデバイスを返す

        Returns:
            device: デバイス情報
        """
        return self.linear.weight.device
