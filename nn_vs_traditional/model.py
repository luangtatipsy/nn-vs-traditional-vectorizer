from typing import Tuple

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC

from nn_vs_traditional.image import ImageReader
from nn_vs_traditional.preprocessing import ImagePreprocessor
from nn_vs_traditional.vectorization import HogVectorizer, ResNet50Vectorizer


def create_nn_based_model(image_size: Tuple[int, int]) -> Pipeline:
    return make_pipeline(
        make_pipeline(
            ImageReader(),
            ImagePreprocessor(image_size=image_size),
            ResNet50Vectorizer(),
        ),
        LinearSVC(),
    )


def create_traditional_based_model(image_size: Tuple[int, int]) -> Pipeline:
    return make_pipeline(
        make_pipeline(
            ImageReader(),
            ImagePreprocessor(image_size=image_size),
            HogVectorizer(),
            LinearSVC(),
        )
    )
