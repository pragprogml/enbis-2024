<h1 align="center">
	<img width="600"
		src="media/enbis-logo.png">
</h1>

<h3 align="center">
  The Anatomy of a Machine Learning Pipeline
</h3>

## Overview

This repository contains the code of the webinar [The Anatomy of a Machine Learning Pipeline](https://conferences.enbis.org/event/49/) based on the book book [The Pragmatic Programmer for Machine Learning](https://www.taylorfrancis.com/books/mono/10.1201/9780429292835/pragmatic-programmer-machine-learning-marco-scutari-mauro-malvestio) published by Taylor & Francis.

## Slides

[The Anatomy of a Machine Learning Pipeline](slides/slides.pdf)

## Prerequisites

To utilize this repository effectively, it's essential to install specific software dependencies on your computer using your Linux distribution's package manager, `brew` for MacOS, or `Chocolately` for Windows. These dependencies are crucial for the optimal operation of the enbis-2024 software stack.

* Docker 20.10+ - [Docker documentation](https://docs.docker.com/get-docker/).
* Docker Compose 2.15+ - [Docker Compose documentation](https://docs.docker.com/compose/install/).
* Python 3.10+ - [Python documentation](https://www.python.org/downloads/).
* VirtualEnv 20.23+ - [VirtualEnv documentation](https://virtualenv.pypa.io/en/latest/installation.html).
* GNU Make 3.81+ - [GNU Make documentation](https://www.gnu.org/software/make/).
* Poetry 1.1.11+ - [Poetry documentation](https://python-poetry.org/docs/#installation).

To verify that you have the correct version of the software installed, run the following commands (for MacOS
and GNU/Linux users):

```sh
(host) $ make check-deps
```

## First installation

```sh
(host) $ git clone https://github.com/pragprogml/enbis-2024
(host) $ cd enbis-2024
(host) $ make venv
(host) $ source .venv/bin/activate
(host) $ pip install poetry
(host) $ poetry install --no-root
(host) $ petry env info

```

## Environment variables

```sh
(host) $ cp .env.example .env
(host) $ vim .env
(host) $ cat .env
ROOT_DIR="/home/user/development/enbis-2024"
PARAMS="params-dev.yaml"
```

## Model training and evaluation

```sh
(host) $ dvc stage list
train     Outputs models/best_model.pt
evaluate  Outputs reports/yolo_metrics.png, reports/predicted_images.png
(host) $ dvc dag
  +-------+
  | train |
  +-------+
      *
      *
      *
+----------+
| evaluate |
+----------+
(host) $ dvc repro train
(host) $ dvc repro evaluate
(host) $ # without previous strage finalization
(host) $ dvc repro --downstream evaluate
```

## Artifacts

The output and artifacts of each training run are stored under `enbis-2024-runs/`, while a straightforward evaluation output can be found in the `reports/` directory.

## Model training and evaluation using Jupyter Notebook

```sh
00_preparations.ipynb
01_visualize.ipynb
01_visualize_iou.ipynb
02_training.ipynb
03_evaluate.ipynb
03_evaluate_onnx.ipynb
03_test_inference_api.ipynb
```

## Experiment Tracking

Update the YOLO setting in ```training.py:training_config``` to enable W&B or MLFlow.

## Evaluation Dashboard UI

```sh
(host) $ make demo
```

## Inference API
```sh
(host) $ make api #for local development
(host) $
(host) $ docker build -t enbis-2024-api -f src/api/Dockerfile .
(host) $ docker run -p 9090:9090 -ti enbis-2024-api
```

## Lint and format

```sh
(host) $ ruff check src/ notebooks/* --fix
(host) $ ruff format src/ notebooks/*
```

## Dataset

The dataset we presented during the seminar is proprietary and unfortunately cannot be shared or used outside of our organization,
you might consider using a dataset that contains a single class, such as the one available at [Synthetic Corrosion Dataset Computer Vision Project](https://universe.roboflow.com/synthetic-corrosion/synthetic-corrosion-dataset), please let us know if you have any questions or need further assistance in finding suitable datasets for your work.

## Contact

Reach out to us via [https://github.com/pragprogml](https://github.com/pragprogml) email.

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
