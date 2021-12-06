# Traditional vs Neural Network Feature Vectorization

# Datasets
- [vechicle-dataset](https://www.kaggle.com/rishabkoul1/vechicle-dataset)

## Prerequisites
- Git
- Python 3.8.7

## Setup
0. Clone the repository
```sh
git clone https://github.com/luangtatipsy/nn-vs-traditional-vectorizer.git
cd nn-vs-traditional-vectorizer
```
1. Create and activate a virtual environment for Python _(recommended)_. If you do not prefer using a virtual environment, skip to step 4.
```sh
python -m venv env
source env/bin/activate
```
2. Update pip to latest version
```sh
python -m pip install --upgrade pip
```
3. Install requirements
```sh
python -m pip install -r requirements.txt
```
4. Download the dataset (as mentioned above) then, unzip an archive file into `datasets` directory (`vechicles` should be inside `datasets`)

5. Run `prepare_dataset.py` to change test images coresponding to classes in the training set.
```sh
python prepare_dataset.py
```
