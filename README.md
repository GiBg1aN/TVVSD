# Visual Verb Disambiguation Game (DRAFT)

This repository contains the code to run the experiments presented in
the paper *Transductive Visual Verb Sense Disambiguation* which will be
published on WACV2021 (link still not available).

## Installation

The project is runs on Python 3.6 and relies on the following packages:

- pytorch 1.0.0
- torchvision 0.2.1
- pandas 0.24.2
- numpy 1.17.3

### Conda environment
Required packages can be easily installed using the conda YAML file:
```
conda env create --file conda_env.yml
```

### Required data

The data can be downloaded from the following link <> and must be placed
in the project folder.

## Experiments

Running the experiments script without any optional parameter will use
default ones, running the transduction on *VerSe dataset* using as
input a data-point and a verb, as described in the paper.
The number of labelled data-points per class will change from 1 to 20.
```
python -m run_experiments
```
The annotations used are the ones from COCO (GOLD).

### Parameters
Optional parameters can be specified:
```
usage: run_experiments.py [-h] [-G | -P] [-m MAX_LABELS] [-a]

optional arguments:
  -h, --help            show this help message and exit
  -G, --gold_captions   Use GOLD captions from COCO.
  -P, --pred_captions   use PRED captions extracted with NeuralBabyTalk.
  -m MAX_LABELS, --max_labels MAX_LABELS
                        The maximum number of labeled data-points to use for
                        each sense.
  -a, --all_senses      Ignore input verb, run inference on the senses of all
                        verbs for each data point.
```

## Inference on your own dataset
TODO
