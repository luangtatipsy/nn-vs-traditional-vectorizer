import os
from typing import List, Tuple


def load_dataset(directory: str) -> Tuple[List[str], List[str]]:
    X = []
    y = []

    for animal in sorted(os.listdir(directory)):
        animal_dir = os.path.join(directory, animal)
        img_file_names = os.listdir(animal_dir)

        for fn in sorted(img_file_names):
            img_path = os.path.join(animal_dir, fn)
            X.append(img_path)
        y.extend([animal] * len(img_file_names))

        assert len(X) == len(y)

    return X, y
