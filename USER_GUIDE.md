# USER_GUIDE

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install `torch` and `torchvision` separately first, using the version appropriate for your machine. Then install the remaining dependencies from `requirements.txt`.

Example for a generic installation:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

Example for CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Download the dataset

Download only the reality subset:

```bash
python scripts/download_dataset.py --real
```


## Create the train/validation split

```bash
python scripts/create_val_split.py
```

This writes the split files to `.local/splits/`.

## Train a model

Small U-Net:

```bash
python scripts/train.py --model small_unet
```

Fast-SCNN:

```bash
python scripts/train.py --model fast_scnn
```

Checkpoints are saved to `.local/checkpoints/`, and training plots are saved to `.local/training_plots/`.

## Run the test script after training

```bash
python scripts/test.py
```

The test script always evaluates both models. It saves metrics, CPU timing, and prediction figures to `.local/test_results/`.

## Run the test script without training

If you want to use the checkpoints already committed in `reports/`, copy them to `.local/checkpoints/` first:

```bash
mkdir -p .local/checkpoints
cp reports/checkpoints/*.pth .local/checkpoints/
python scripts/test.py
```

The script expects:

- `.local/checkpoints/small_unet_best.pth`
- `.local/checkpoints/fast_scnn_best.pth`
