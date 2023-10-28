import argparse

from vision.datasets.base import DatasetBase
from torchinfo import summary  # noqa

from .base import ModelBase
from .multi_class_logistic_regression import MultiClassLogisticRegression
from .feedforward_neural_network import FNN
from .resnet18 import ResNet18


def generate_model(args: argparse.Namespace, dataset: DatasetBase) -> ModelBase:
    """モデルを生成する

    Args:
        args (argparse.Namespace): コンソール引数
        dataset (DatasetBase): データセット

    Raises:
        NotImplementedError: 実装されていないモデル名を指定した場合

    Returns:
        ModelBase: モデル
    """
    if args.model == "mlr":
        model = MultiClassLogisticRegression(args.dim_input, dataset.num_classes)
    elif args.model == "fnn":
        model = FNN(
            args.dim_input, args.dim_hidden, args.num_hidden_layers, dataset.num_classes
        )
        model.to(args.device)
    elif args.model == "resnet18":
        model = ResNet18(dataset.num_classes)
        model.to(args.device)
    else:
        raise NotImplementedError("実装されていないモデル名です")

    summary(model, input_size=(32, 3, 224, 224))

    return model
