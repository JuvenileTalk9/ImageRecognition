import numpy as np

from .base import MlModelBase


class MultiClassLogisticRegression(MlModelBase):
    """多クラスロジスティック回帰"""

    def __init__(self, dim_input: int, num_classes: int) -> None:
        """コンストラクタ

        Args:
            dim_input (int): 入力の次元数
            num_classes (int): クラス数
        """
        self.weight = np.random.normal(scale=0.01, size=(dim_input, num_classes))
        self.bias = np.zeros(num_classes)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """各クラスの確率を予測する

        Args:
            x (np.ndarray): 画像データ

        Returns:
            np.ndarray: 各クラスの確率
        """
        logit = np.matmul(x, self.weight) + self.bias
        y = MlModelBase.softmax(logit)
        return y

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
        diffs = y_pred - y
        self.weight -= lr * np.mean(x[:, :, np.newaxis] * diffs[:, np.newaxis], axis=0)
        self.bias -= lr * np.mean(diffs, axis=0)
