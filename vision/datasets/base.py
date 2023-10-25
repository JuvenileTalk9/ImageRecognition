import random
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DatasetBase:
    """データセットのラッパークラスの基底クラス"""

    def __init__(self) -> None:
        """コンストラクタ"""
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.num_classes: int = None

    @classmethod
    def generate_subset(
        cls,
        dataset: Dataset,
        ratio: float,
        seed: int = 0,
    ) -> Tuple[list[int], list[int]]:
        """データセットを2つに分割するインデックス集合を生成

        Args:
            dataset (Dataset): データセット
            ratio (float): 1つめの集合に含めるデータ量の割合
            seed (int, optional): 乱数のシード値

        Returns:
            Tuple[list[int], list[int]]: インデックス集合のタプル
        """
        size = int(len(dataset) * ratio)
        indices = list(range(len(dataset)))

        random.seed(seed)
        random.shuffle(indices)

        indices1, indices2 = indices[:size], indices[size:]
        return indices1, indices2

    @classmethod
    def get_dataset_statistics(cls, dataset: Dataset) -> Tuple[float, float]:
        """データセット全体の平均と標準偏差を計算する

        Args:
            dataset (Dataset): データセット

        Returns:
            Tuple[float, float]: 平均と標準偏差
        """
        imgs = []
        for data in dataset:
            imgs.append(data[0])
        imgs = np.stack(imgs)

        channel_mean = np.mean(imgs, axis=0)
        channel_std = np.std(imgs, axis=0)

        return channel_mean.flatten(), channel_std.flatten()

    @classmethod
    def standardization(
        cls,
        img: Image.Image,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
    ) -> np.ndarray:
        """画像を標準化する

        Args:
            img (Image.Image): 整形対象の画像
            channel_mean (np.ndarray): データセットの平均
            channel_std (np.ndarray): データセット全体の標準偏差

        Returns:
            np.ndarray: 標準化後の画像
        """
        x = DatasetBase.to_flatten(img)
        x = (x - channel_mean) / channel_std
        return x

    @classmethod
    def to_flatten(cls, img: Image.Image) -> np.ndarray:
        """画像を平滑化する

        Args:
            img (Image.Image): 整形対象の画像

        Returns:
            np.ndarray: 平滑化後の画像
        """
        img_np = np.asarray(img, dtype="float32")
        x = img_np.flatten()
        return x

    @classmethod
    def to_one_hot_expression(cls, label: int, num_classes: int) -> np.ndarray:
        """ラベルをOne-hot表現に変換する

        Args:
            label (int): ラベル番号
            num_classes (int): クラス数

        Returns:
            np.ndarray: One-hot表現のラベル
        """
        return np.identity(num_classes)[label]
