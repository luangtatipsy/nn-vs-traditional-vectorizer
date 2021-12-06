import argparse
import os
import time
from typing import List

import joblib
import psutil
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from utils.helper import load_dataset


def predict(pipeline: Pipeline, X_test: List[str]) -> List[str]:
    return pipeline.predict(X_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pipeline",
        type=str,
        help="Which pipeline being used to predict images in a test set (nn/traditional)",
    )
    parser.add_argument(
        "--image_size",
        default=256,
        type=int,
        help="image size",
    )

    args = parser.parse_args()

    if args.pipeline == "nn":
        pipeline = joblib.load(os.path.join("models", "nn.pipeline"))
    elif args.pipeline == "traditional":
        pipeline = joblib.load(os.path.join("models", "traditional.pipeline"))
    else:
        raise NotImplementedError("`pipeline` must be one of `nn` or `traditional`")

    TEST_DIR = os.path.join("datasets", "vechicles", "test")
    X_test, y_test = load_dataset(TEST_DIR)

    print(f"Start predicting {args.pipeline} pipeline...")
    process = psutil.Process(os.getpid())
    start_time = time.time()
    y_pred = predict(pipeline, X_test)
    end_time = time.time() - start_time

    print(f"Execution time: {round(end_time, 2)} seconds")
    print(f"CPU Usage: {round(psutil.cpu_percent(), 2)}%")
    print(f"Memory Usage: {round(process.memory_percent(), 2)}%")
    print()
    print(classification_report(y_pred=y_pred, y_true=y_test, digits=4))
