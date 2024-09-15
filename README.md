
<h1 align="center">The Anatomy of a Machine Learning Pipeline</h1>

<p align="center">
  <img width="190px" src="media/enbis-logo-s.png" style="display: inline-block; vertical-align: middle; margin-right: 20px;">
  <img width="165px" src="media/bicocca-logo.png" style="display: inline-block; vertical-align: middle; margin-left: 20px;">
</p>

## Overview

This repository contains the code of the webinar The Anatomy of a Machine
Learning Pipeline, presented both at ENBIS and at the University of
Milan-Bicocca, based on the book [The Pragmatic Programmer for Machine
Learning](https://www.taylorfrancis.com/books/mono/10.1201/9780429292835/pragmatic-programmer-machine-learning-marco-scutari-mauro-malvestio)
published by Taylor & Francis.

## Editions

* [16th September 2024 - Università degli Studi di Milano-Bicocca](https://www.unimib.it/eventi/anatomy-machine-learning-pipeline) 
* [10th January 2024 - European Network for Business and Industry Statistics](https://conferences.enbis.org/event/49/)

## Slides

[The Anatomy of a Machine Learning Pipeline](slides/slides.pdf)

## Prerequisites

To utilize this repository effectively, it's essential to install specific
software dependencies on your computer using your Linux distribution's package
manager, `brew` for MacOS, or `Chocolately` for Windows. These dependencies are
crucial for the optimal operation of the codebase and reproducibility.

* Docker 26+ - [Docker documentation](https://docs.docker.com/get-docker/).
* Docker Compose 2.27+ - [Docker Compose documentation](https://docs.docker.com/compose/install/).
* Python 3.10+ - [Python documentation](https://www.python.org/downloads/).
* Poetry 1.8+ - [Poetry documentation](https://python-poetry.org/docs/#installation).
* GNU Make 3.81+ - [GNU Make documentation](https://www.gnu.org/software/make/).

To verify that you have the correct version of the software installed, run the
following commands (for MacOS and GNU/Linux users):

```sh
make check-deps
```

## First installation

We suggest using `asdf` to manage the Python version, you can install it
following the instructions at [asdf](https://asdf-vm.com/#/core-manage-asdf-vm).

```sh
asdf install plugin python
asdf install python 3.10.15
asdf plugin-add direnv 
asdf direnv setup --shell bash --version latest 
python --version
pip install poetry
poetry config virtualenvs.in-project true
poetry config virtualenvs.path .venv
git clone https://github.com/pragprogml/enbis-2024
cd enbis-2024
poetry install --no-root
poetry env info
```

## Environment variables

```sh
cp .envrc.example .envrc
vim .envrc
cat .envrc
```

```sh
layout python
ROOT_DIR="/home/user/development/enbis-2024"
PARAMS="params-dev.yaml"
VIRTUAL_ENV=.venv
PATH=$VIRTUAL_ENV/bin:$PATH
```

## Model training and evaluation

```sh
dvc stage list
```

```
train     Outputs models/best_model.pt
evaluate  Outputs reports/yolo_metrics.png, reports/predicted_images.png
```

```sh
dvc dag
```

```
  +-------+
  | train |
  +-------+
      *
      *
      *
+----------+
| evaluate |
+----------+
```

```sh
dvc repro train
dvc repro evaluate
dvc repro --downstream evaluate # without previous stage finalization
```

## Artifacts

The output and artifacts of each training run are stored under
`runs/`, while a straightforward evaluation output can be found in
the `reports/` directory.

## Model training and evaluation using Jupyter Notebook

* [00_preparations.ipynb](notebooks/00_preparations.ipynb)
* [01_visualize.ipynb](notebooks/01_visualize.ipynb)
* [01_visualize_iou.ipynb](notebooks/01_visualize_iou.ipynb)
* [02_training.ipynb](notebooks/02_training.ipynb)
* [03_evaluate.ipynb](notebooks/03_evaluate.ipynb)
* [03_evaluate_onnx.ipynb](notebooks/03_evaluate_onnx.ipynb)
* [03_inference_api.ipynb](notebooks/03_inference_api.ipynb)

## Experiment Tracking

Update the YOLO setting in `training.py:training_config` to enable W&B or MLFlow, or use 

```sh
yolo settings mlflow={True|False} wandb={True|False}
```

## Web Evaluation Dashboard 

```sh
make run-demo
```

## Inference API

```sh
make run-api

make docker-build
make docker-run
```

## Lint and format

```sh
make lint
```

## Dataset

The dataset we presented during the seminar is proprietary and unfortunately
cannot be shared or used outside of our organization. You might consider using a
dataset that contains a single class, such as the one available at [Synthetic
Corrosion Dataset Computer Vision
Project](https://universe.roboflow.com/synthetic-corrosion/synthetic-corrosion-dataset). Please let us know if you have any questions or need further assistance in
finding suitable datasets for your work.

## Contact

Reach out to us via [https://github.com/pragprogml](https://github.com/pragprogml).

## Authors

- [Marco Scutari, Ph.D.](https://www.bnlearn.com/) - Senior Researcher in Bayesian Networks and Graphical Models - Istituto Dalle Molle di Studi sull'Intelligenza Artificiale (IDSIA)
- [Mauro Malvestio](https://linktr.ee/mauro.malvestio) - Founder & CTO @ DSCOVR 

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Citing

When citing PPML in academic papers and theses, please use this BibTeX entry:

```bibtex
@BOOK{ppml,
  author        = {M. Scutari and M. Malvestio},
  title         = {{The Pragmatic Programmer for Machine Learning: Engineering
                    Analytics and Data Science Solutions}},
  publisher     = {Chapman \& Hall},
  year          = {2023}
}
```
