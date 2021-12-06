from typing import List, Tuple

from PIL import Image
from typing_extensions import Self


class ImageReader:
    def read(self, img_path: str) -> Image.Image:
        return Image.open(img_path)

    def to_rgba(self, img: Image.Image) -> Image.Image:
        return img.convert("RGBA")

    def fill_background(
        self, img: Image.Image, bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        bg_img = Image.new(
            mode="RGBA", size=img.size, color=bg_color
        )  # Create a given background color.
        bg_img.paste(img, (0, 0), img)  # Paste the image on the background.

        return bg_img.convert("RGB")

    def fit(self, X, y=None, **fit_params) -> Self:
        return self

    def transform(self, X: List[str]) -> List[Image.Image]:
        imgs = []
        for img_path in X:
            img = self.read(img_path)
            if img.mode != "RGB":
                if img.mode == "P":
                    img = self.to_rgba(img)
                img = self.fill_background(img)
            imgs.append(img)

        return imgs
