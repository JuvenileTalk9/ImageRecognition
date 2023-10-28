import sys
import logging
import logging.config
import argparse
from collections import deque
from typing import Tuple, Callable

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from vision.datasets import generate_dataset, DatasetBase
from vision.models import generate_model
from vision.models.base import ModelType, MlModelBase, DnnModelBase
from vision.settings import LOGGING_CONFIG


logger = logging.getLogger("image_recognition")


def main(argv: list[str]) -> None:
    """メイン関数

    Args:
        argv (list[str]): コマンドライン引数
    """

    # ログ設定
    logging.config.dictConfig(LOGGING_CONFIG)

    # 引数のパース
    parser = get_parser()
    args = parser.parse_args()

    # データセット
    dataset = generate_dataset(root="data/datasets", args=args)

    # モデル
    model = generate_model(args, dataset)

    if model.get_model_type() == ModelType.ML:
        # 機械学習モデルの学習・評価
        train_eval_ml(args, dataset, model)
    elif model.get_model_type() == ModelType.DNN:
        # DNNモデルの学習評価
        train_eval_dnn(args, dataset, model)
    else:
        raise NotImplementedError("存在しないモデルタイプです")


def get_parser() -> argparse.ArgumentParser:
    """引数のパーサを生成する

    Returns:
        argparse.ArgumentParser: パーサ
    """
    parser = argparse.ArgumentParser()

    # モデル名
    parser.add_argument(
        "-m", "--model", type=str, required=True, choices=["mlr", "fnn", "resnet18"]
    )
    # データセット名
    parser.add_argument(
        "-d", "--dataset", type=str, default="cifar10", choices=["cifar10"]
    )
    # 学習パラメータ
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--moving_avg", type=int, default=20)
    # モデルパラメータ
    parser.add_argument("--dim_input", type=int, default=32 * 32 * 3)
    parser.add_argument("--dim_hidden", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    # デバイス
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    return parser


def train_eval_ml(
    args: argparse.Namespace, dataset: DatasetBase, model: MlModelBase
) -> None:
    def evaluate(data_loader: DataLoader, model: MlModelBase) -> Tuple[float, float]:
        """評価する

        Args:
            data_loader (DataLoader): データローダ
            model (MlModelBase): モデル

        Returns:
            Tuple[float, float]: 誤差と正確度
        """
        losses = []
        preds = []
        for x, y in data_loader:
            x = x.numpy()
            y = y.numpy()

            y_pred = model.predict(x)

            losses.append(np.sum(-y * np.log(y_pred), axis=1))
            preds.append(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

        loss = np.mean(np.concatenate(losses))
        accuracy = np.mean(np.concatenate(preds))

        return loss, accuracy

    for epoch in range(args.epoch):
        with tqdm(dataset.train_loader) as pbar:
            pbar.set_description(f"[Epoch {epoch + 1}]")

            losses = deque()
            accs = deque()
            for x, y in pbar:
                x = x.numpy()
                y = y.numpy()
                # 順伝搬
                y_pred = model.predict(x)
                # 誤差と正確度を計算
                loss = np.mean(np.sum(-y * np.log(y_pred), axis=1))
                accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                # 表示用の誤差と正確度を計算
                losses.append(loss)
                accs.append(accuracy)
                if len(losses) > args.moving_avg:
                    losses.popleft()
                    accs.popleft()
                pbar.set_postfix({"loss": np.mean(losses), "accuracy": np.mean(accs)})
                # パラメータの更新
                model.optimize(x, y, y_pred, args.lr)

        # 検証
        val_loss, val_accuracy = evaluate(dataset.val_loader, model)
        logger.info(f"検証: loss = {val_loss:.3f}, " f"accuracy = {val_accuracy:.3f}")

    # テスト
    test_loss, test_accuracy = evaluate(dataset.test_loader, model)
    logger.info(f"テスト: loss = {test_loss:.3f}, " f"accuracy = {test_accuracy:.3f}")


def train_eval_dnn(
    args: argparse.Namespace, dataset: DatasetBase, model: DnnModelBase
) -> None:
    def evaluate(
        data_loader: DataLoader, model: DnnModelBase, loss_func: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """評価する

        Args:
            data_loader (DataLoader): データローダ
            model (MlModelBase): モデル
            loss_func (Callable): 誤差関数

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 誤差と正確度
        """
        model.eval()

        losses = []
        preds = []
        for x, y in data_loader:
            with torch.no_grad():
                x = x.to(model.get_device())
                y = y.to(model.get_device())

                y_pred = model(x)

                losses.append(loss_func(y_pred, y, reduction="none"))
                preds.append(y_pred.argmax(dim=1) == y)

        loss = torch.cat(losses).mean()
        accuracy = torch.cat(preds).float().mean()

        return loss, accuracy

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_func = model.get_loss_function()

    for epoch in range(args.epoch):
        model.train()

        with tqdm(dataset.train_loader) as pbar:
            pbar.set_description(f"[Epoch {epoch + 1}]")

            losses = deque()
            accs = deque()
            for x, y in pbar:
                x = x.to(model.get_device())
                y = y.to(model.get_device())

                # 勾配をリセット
                optimizer.zero_grad()
                # 順伝搬
                y_pred = model(x)
                # 誤差と正確度を計算
                loss = loss_func(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == y).float().mean()
                # 誤差逆伝搬
                loss.backward()
                # パラメータの更新
                optimizer.step()
                # 表示用の誤差と正確度を計算
                losses.append(loss.item())
                accs.append(accuracy.item())
                if len(losses) > args.moving_avg:
                    losses.popleft()
                    accs.popleft()
                pbar.set_postfix(
                    {
                        "loss": torch.Tensor(losses).mean().item(),
                        "accuracy": torch.Tensor(accs).mean().item(),
                    }
                )

        # 検証
        val_loss, val_accuracy = evaluate(dataset.val_loader, model, loss_func)
        logger.info(f"検証: loss = {val_loss:.3f}, " f"accuracy = {val_accuracy:.3f}")

    # テスト
    test_loss, test_accuracy = evaluate(dataset.test_loader, model, loss_func)
    logger.info(f"テスト: loss = {test_loss:.3f}, " f"accuracy = {test_accuracy:.3f}")


if __name__ == "__main__":
    main(sys.argv)
