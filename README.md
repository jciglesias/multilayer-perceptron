# multilayer-perceptron

Simple educational multilayer perceptron implementation in pure Python.

This repository contains a small neural-network implementation and helper
scripts to split a CSV dataset, train a model, and run inference on a test
set. The pipeline is intentionally minimal so you can run everything with
plain Python and a few common scientific packages.

## Repository structure

- `data/` - input dataset and split files: `data.csv`, `training.csv`,
  `validation.csv`, `test.csv`.
- `model.pkl` - example saved model (created after training).
- `metrics.png` - example training metrics plot (created after training).
- `src/` - source code
  - `split_data.py` - create `training.csv`, `validation.csv`, `test.csv` from `data/data.csv`
  - `train_neuralnetwork.py` - train a `NeuralNetwork` and save it to `model.pkl`
  - `multilayer_perceptron.py` - run the trained model on `data/test.csv` and print metrics
  - `neuralnetwork.py`, `neuron.py` - core network implementation

## Quick summary

- Inputs: `data/data.csv` (rows: id, label, 30 numeric features)
- Outputs: `data/training.csv`, `data/validation.csv`, `data/test.csv`, `model.pkl`, `metrics.png`
- Success: training finishes and `model.pkl` & `metrics.png` are created; inference prints test differences and loss

## Requirements

- Python 3.8+ (3.10/3.12 tested in this project)
- The following Python packages:

  - pandas
  - numpy
  - matplotlib

Install dependencies (recommended inside a virtualenv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy matplotlib
```

## Data format

The code expects `data/data.csv` with no header. Each row must follow this
layout:

1. id (ignored by the scripts)
2. label: `M` (malignant) or `B` (benign)
3. 30 numeric feature columns used as inputs to the network

An example row (first lines of `data/data.csv` are included in the repo):

```
842302,M,17.99,10.38,... (30 feature columns)
```

## How to run

Run these three steps in order from the repository root.

1) Split the data

This creates `data/training.csv`, `data/validation.csv`, and `data/test.csv` from `data/data.csv`.

```bash
python3 src/split_data.py
```

2) Train the model

Trains a NeuralNetwork using `data/training.csv` and `data/validation.csv`,
saves the trained model to `model.pkl`, and writes a `metrics.png` plot.

```bash
python3 src/train_neuralnetwork.py
```

3) Run inference / evaluate

This loads `model.pkl` and evaluates it on `data/test.csv`. The script prints
the number of differences and a (binary cross-entropy) loss value.

```bash
python3 src/multilayer_perceptron.py
```

Notes:
- If any of the expected files are missing the scripts will print `File not found` and exit.
- `train_neuralnetwork.py` constructs a `NeuralNetwork(30, 40, 4)` by default
  (30 inputs, 40 neurons per hidden layer, 4 hidden layers) and calls
  `train(..., learning_rate=0.001, epochs=100)` — edit the file to change
  hyperparameters.

## Outputs

- `model.pkl` — the pickled NeuralNetwork after training
- `metrics.png` — training/validation loss and accuracy vs epochs

## Troubleshooting

- If plots don't appear when training, ensure a display is available or
  run headless by removing `plt.show()` in `src/neuralnetwork.py` (the file
  still writes `metrics.png`).
- If you see `ModuleNotFoundError` for any package, install it via pip as
  shown in Requirements.

## License

See the `LICENSE` file in the repository for license terms.