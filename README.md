# Python coding good practices

This repository is an example of python code that uses folders and files to organize the code.

## Installation

First, you need to create a python environment. You can use `pyenv`, `conda`, `poetry`, etc.

For instance, you can create a `conda` environment using:

```bash
conda create -n my_env python=3.10 -y 
```

Then, you can activate the environment:

```bash
conda activate my_env
```

Then, you can install the requirements using, for instance `pip`:
```bash
pip install -r requirements.txt
```

## Usage

To run the code that is in `src/training/training_helper.py`, you can use:

```bash
python -m src.training.training_helper
```

Note the that `/` are replaced by `.` and the `.py` extension is removed.

You can have a look at the importation format in the different files in the `src` folder. 

You need to run the code from the root of the repository, otherwise the imports will not work.