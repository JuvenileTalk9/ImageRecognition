from typing import Any, Callable

from .base import DnnModelBase

from torch import nn, Tensor, device


class FNN(DnnModelBase):
    """順伝搬型ニューラルネットワーク"""

    def __init__(
        self, dim_input: int, dim_hidden: int, num_hidden_layers: int, num_classes: int
    ) -> None:
        """コンストラクタ

        Args:
            dim_input (int): 入力データの次元数
            dim_hidden (int): 隠れ層の次元数
            num_hideen_layers (int): 隠れ層の数
            num_classes (int): 分類クラス数
        """
        super().__init__()
        self.dim_input = dim_input
        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        self.layers.append(self.__generate_hidden_layer(dim_input, dim_hidden))

        for _ in range(num_hidden_layers - 1):
            self.layers.append(self.__generate_hidden_layer(dim_hidden, dim_hidden))

        self.linear = nn.Linear(dim_hidden, num_classes)

    def forward(self, x: Tensor, return_embed: bool = False) -> Any:
        """順伝搬関数

        Args:
            x (Tensor): 入力
            return_embed (bool, optional): 真なら特徴量、偽ならロジットを返すフラグ

        Returns:
            Any: 真なら特徴量、偽ならロジット
        """
        for layer in self.layers:
            x = layer(x)

        # return_embedが真の場合特徴量を返す
        if return_embed:
            return x
        # return_embedが偽の場合特徴量を返す
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

    def __generate_hidden_layer(self, dim_input: int, dim_output: int) -> nn.Sequential:
        """隠れ層を生成する

        Args:
            dim_input (int): 入力の次元数
            dim_output (int): 出力の次元数

        Returns:
            nn.Sequential: 隠れ層
        """
        layer = nn.Sequential(
            nn.Linear(dim_input, dim_output, bias=False),
            nn.BatchNorm1d(dim_output),
            nn.ReLU(inplace=True),
        )
        return layer
