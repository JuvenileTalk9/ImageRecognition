from typing import Any, Callable

from .base import DnnModelBase

import torch


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
        super().__init__(dim_input, num_classes)
        self.layers = torch.nn.ModuleList()

        self.layers.append(self.__generate_hidden_layer(dim_input, dim_hidden))

        for _ in range(num_hidden_layers - 1):
            self.layers.append(self.__generate_hidden_layer(dim_hidden, dim_hidden))

        self.linear = torch.nn.Linear(dim_hidden, num_classes)

    def forward(self, x: torch.Tensor, return_embed: bool = False) -> Any:
        """順伝搬関数

        Args:
            x (torch.Tensor): 入力
            return_embed (bool, optional): 真なら特徴量、偽ならロジットを返すフラグ

        Returns:
            Any: 真なら特徴量、偽ならロジット
        """
        h = x
        for layer in self.layers:
            h = layer(h)

        # return_embedが真の場合特徴量を返す
        if return_embed:
            return h
        # return_embedが偽の場合特徴量を返す
        y = self.linear(h)
        return y

    def get_loss_function(self) -> Callable:
        """誤差関数を返す"""
        return torch.nn.functional.cross_entropy

    def get_device(self) -> torch.device:
        """重みを保持しているデバイスを返す

        Returns:
            device: デバイス情報
        """
        return self.linear.weight.device

    def __generate_hidden_layer(
        self, dim_input: int, dim_output: int
    ) -> torch.nn.Sequential:
        """隠れ層を生成する

        Args:
            dim_input (int): 入力の次元数
            dim_output (int): 出力の次元数

        Returns:
            torch.nn.Sequential: 隠れ層
        """
        layer = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_output, bias=False),
            torch.nn.BatchNorm1d(dim_output),
            torch.nn.ReLU(inplace=True),
        )
        return layer
