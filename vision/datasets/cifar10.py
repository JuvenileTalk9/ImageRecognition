import logging

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .base import DatasetBase


logger = logging.getLogger("image_recognition")


class CIFAR10(DatasetBase):
    """CIFAR10のラッパークラス"""

    def __init__(
        self,
        root: str,
        val_ratio: int,
        batch_size: int,
        num_workers: int,
        enable_flatten: bool = False,
        enable_one_hot_expression: bool = False,
    ) -> None:
        """コンストラクタ

        Args:
            root (str): ダウンロードファイルを保存するルートディレクトリ
            val_ratio (int): 学習データセット内の検証に使うデータの割合
            batch_size (int): バッチサイズ
            num_workers (int): データローダーに使うCPUプロセスの数
            enable_flatten (bool, optional): 真なら画像を一次元化する
            enable_one_hot_expression (bool, optional): ラベルをOne-hot表現に変換するか否か
        """
        super().__init__()

        if enable_flatten:
            transform = DatasetBase.to_flatten
        else:
            transform = T.ToTensor()

        # 学習用のデータセットを取得
        train_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=transform,
        )

        # 平均と分散を計算
        channel_mean, channel_std = DatasetBase.get_dataset_statistics(
            train_dataset, enable_flatten
        )
        if enable_flatten:
            # transformのラムダ式
            img_transform = lambda x: DatasetBase.standardization(  # noqa
                x, channel_mean, channel_std
            )
        else:
            img_transform = T.Compose(
                (
                    T.ToTensor(),
                    T.Normalize(mean=channel_mean, std=channel_std),
                )
            )

        # クラス数を取得
        self.num_classes = len(train_dataset.classes)

        if enable_one_hot_expression:
            label_transform = lambda y: DatasetBase.to_one_hot_expression(  # noqa
                y, self.num_classes
            )
        else:
            label_transform = None

        # データセットを取得
        train_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=img_transform,
            target_transform=label_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=img_transform,
            target_transform=label_transform,
        )

        # 学習用と検証用のサブセットを生成
        val_set, train_set = DatasetBase.generate_subset(train_dataset, val_ratio)

        logger.debug(f"学習セットのサンプル数: {len(train_set)}")
        logger.debug(f"検証セットのサンプル数: {len(val_set)}")
        logger.debug(f"テストセットのサンプル数: {len(test_dataset)}")

        # サブセットをもとにランダムサンプラを生成
        train_sampler = SubsetRandomSampler(train_set)
        val_sampler = SubsetRandomSampler(val_set)

        # DataLoaderを生成
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
        )
        self.val_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=val_sampler,
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers
        )
