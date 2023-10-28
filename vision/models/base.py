import copy
from typing import Any, Callable
from enum import Enum

import numpy as np
from torch import nn, device


class ModelType(Enum):
    """モデルタイプを表す列挙型"""

    ML: str = 1
    DNN: str = 2


class ModelBase:
    """モデルのベースクラス"""

    def get_model_type(self) -> str:
        """モデルの種別を返す

        Returns:
            str: モデルの種別
        """
        raise NotImplementedError

    def copy(self) -> Any:
        """モデルを複製する

        Returns:
            Any: 複製されたモデル
        """
        return copy.deepcopy(self)


class MlModelBase(ModelBase):
    """機械学習モデルのベースクラス"""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """各クラスの確率を予測する"""
        raise NotImplementedError

    def optimize(
        self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, lr: float = 0.001
    ) -> None:
        """パラメータを更新する

        Args:
            x (np.ndarray): 入力データ
            y (np.ndarray): 正解のラベル
            y_pred (np.ndarray): 推測したラベル
            lr (float, optional): 学習率
        """
        raise NotImplementedError

    def get_model_type(self) -> str:
        """モデルの種別を返す

        Returns:
            str: モデルの種別
        """
        return ModelType.ML

    @classmethod
    def softmax(cls, x: np.ndarray) -> np.ndarray:
        """ソフトマックス関数

        Args:
            x (np.ndarray): ロジット

        Returns:
            np.ndarray: 関数の出力
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class DnnModelBase(ModelBase, nn.Module):
    """DNNのベースクラス"""

    def __init__(self) -> None:
        """コンストラクタ"""
        super().__init__()

    def forward(self, **kwargs) -> Any:
        """順伝搬関数"""
        raise NotImplementedError

    def get_loss_function(self) -> Callable:
        """誤差関数を返す"""
        raise NotImplementedError

    def get_device(self) -> device:
        """重みを保持しているデバイスを返す"""
        raise NotImplementedError

    def get_model_type(self) -> str:
        """モデルの種別を返す

        Returns:
            str: モデルの種別
        """
        return ModelType.DNN
