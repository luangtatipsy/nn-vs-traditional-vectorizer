import numpy as np
from PIL import Image, ImageFilter
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from typing_extensions import Self


class ResNet50Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.model = ResNet50(
            input_shape=(None, None, 3), include_top=False, weights="imagenet"
        )
        self.preprocessing_fn = preprocess_input

    def __preprocess(self, img: np.ndarray) -> np.ndarray:
        return self.preprocessing_fn(img)

    def __vectorize(self, img: np.ndarray) -> np.ndarray:
        x = self.__preprocess(img)
        x = np.expand_dims(x, axis=0)
        x = self.model(x)
        x = GlobalMaxPool2D()(x)

        return x[0].numpy()

    def fit(self, X, y=None, **fit_params) -> Self:
        return self

    def transform(self, X):
        return np.array([self.__vectorize(img) for img in X])


class TraditionalVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.kernel = ImageFilter.Kernel(
            (3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0
        )  # laplican Kernel

    def __mean_pixel_of_channels(self, img: np.ndarray) -> np.ndarray:
        height, width, _ = img.shape
        feature_vector = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                feature_vector[i][j] = (
                    int(img[i, j, 0]) + int(img[i, j, 1]) + int(img[i, j, 2])
                ) / 3

        return feature_vector

    def __extract_edge(self, img: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.convert("L")  # Convert color mode from RGB to grayscale
        edge_array = np.array(pil_img.filter(self.kernel))

        return edge_array.flatten()

    def __vectorize(self, img: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (self.__mean_pixel_of_channels(img), self.__extract_edge(img)), axis=None
        )

    def fit(self, X, y=None, **fit_params) -> Self:
        return self

    def transform(self, X) -> np.ndarray:
        return np.array([self.__vectorize(img) for img in X])


class HogVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm="L2-Hys",
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None, **fit_params) -> Self:
        return self

    def transform(self, X) -> np.ndarray:
        grayscale_imgs = [Image.fromarray(img).convert("L") for img in X]

        return np.array(
            [
                hog(
                    img,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    block_norm=self.block_norm,
                )
                for img in grayscale_imgs
            ]
        )
