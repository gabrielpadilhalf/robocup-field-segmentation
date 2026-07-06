# robocup-field-segmentation

Binary semantic segmentation project for soccer-field perception with the TORSO-21 reality dataset.

The repository compares two lightweight models:

- `Small U-Net`
- `Fast-SCNN`

The task is binary:

- class `0`: non-field
- class `1`: field

The main goals are:

- compare segmentation quality with IoU and Dice
- compare CPU inference cost

## Directory overview

- `configs/`: base config and model-specific training configs
- `docs/`: architecture notes, implementation backlog, and experiment notes
- `reports/`: versioned artifacts already generated for the project
- `scripts/`: entry-point scripts for dataset download, split creation, training, testing, and transform inspection
- `src/field_segmentation/data/`: dataset loading, split handling, and transforms
- `src/field_segmentation/models/`: `small_unet` and `fast_scnn`
- `src/field_segmentation/train/`: training loop and checkpoint logic
- `src/field_segmentation/eval/`: evaluation metrics
- `src/field_segmentation/utils/`: config loading and utility helpers

## Versioned results

The `reports/` directory contains generated artifacts that can be inspected without rerunning training:

- `reports/checkpoints/`: trained checkpoints
- `reports/test_results/`: test metrics, timing results, and prediction figures
- `reports/training_plots/`: training curves

## More information

- setup and execution: [USER_GUIDE.md](/home/gabriel/ITA/2comp/manga/robocup-field-segmentation/USER_GUIDE.md)
