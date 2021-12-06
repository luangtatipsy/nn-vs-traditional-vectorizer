import argparse
import os
import time
from typing import List

import joblib
import psutil
from sklearn.pipeline import Pipeline

from nn_vs_traditional.model import (
    create_nn_based_model,
    create_traditional_based_model,
)
from utils.helper import load_dataset


def train(pipeline: Pipeline, X_train: List[str], y_train: List[str]) -> None:
    pipeline.fit(X_train, y_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "vectorizer", type=str, help="Which vectorizer being trained (nn/traditional)"
    )
    parser.add_argument(
        "--image_size",
        default=256,
        type=int,
        help="image size",
    )

    args = parser.parse_args()

    if args.vectorizer == "nn":
        pipeline = create_nn_based_model(
            image_size=((args.image_size, args.image_size))
        )
    elif args.vectorizer == "traditional":
        pipeline = create_traditional_based_model(
            image_size=((args.image_size, args.image_size))
        )
    else:
        raise NotImplementedError("`vectorizer` must be one of `nn` or `traditional`")

    TRAIN_DIR = os.path.join("datasets", "vechicles", "train")
    X_train, y_train = load_dataset(TRAIN_DIR)

    print(f"Start training {args.vectorizer} vectorizer...")
    process = psutil.Process(os.getpid())
    start_time = time.time()
    train(pipeline, X_train, y_train)
    end_time = time.time() - start_time

    print(f"Execution time: {round(end_time, 2)} seconds")
    print(f"CPU Usage: {round(psutil.cpu_percent(), 2)}%")
    print(f"Memory Usage: {round(process.memory_percent(), 2)}%")

    joblib.dump(pipeline, os.path.join("models", f"{args.vectorizer}.pipeline"))
