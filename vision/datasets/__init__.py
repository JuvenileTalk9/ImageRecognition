import argparse

from .base import DatasetBase
from .cifar10 import CIFAR10


DATASETS = {
    "cifar": CIFAR10,
}


def generate_dataset(
    root: str,
    args: argparse.Namespace,
) -> DatasetBase:
    """データセットを生成する

    Args:
        root (str): ダウンロードファイルを保存するルートディレクトリ
        args (argparse.Namespace): コンソール引数

    Raises:
        NotImplementedError: 実装されていないデータセット名を指定した場合

    Returns:
        DatasetBase: データセット
    """
    if args.dataset == "cifar10":
        if args.model in ["mlr"]:
            dataset = CIFAR10(
                root,
                args.val_ratio,
                args.batch_size,
                args.num_workers,
                enable_one_hot_expression=True,
            )
        elif args.model in ["fnn"]:
            dataset = CIFAR10(
                root,
                args.val_ratio,
                args.batch_size,
                args.num_workers,
                enable_one_hot_expression=False,
            )
        else:
            raise NotImplementedError("実装されていないモデル名です")
    else:
        raise NotImplementedError("実装されていないデータセット名です")
    return dataset
