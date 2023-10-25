import argparse

from vision.datasets.base import DatasetBase
from .base import ModelBase
from .multi_class_logistic_regression import MultiClassLogisticRegression
from .feedforward_neural_network import FNN


def generate_model(args: argparse.Namespace, dataset: DatasetBase) -> ModelBase:
    if args.model == "mlr":
        model = MultiClassLogisticRegression(args.dim_input, dataset.num_classes)
    elif args.model == "fnn":
        model = FNN(
            args.dim_input, args.dim_hidden, args.num_hidden_layers, dataset.num_classes
        )
        model.to(args.device)
    else:
        raise NotImplementedError("実装されていないモデル名です")

    return model
