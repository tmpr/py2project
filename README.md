dataimputer
==============================

Train a neural network to predict cropped out rectanglular area of an image.

### Remark

This repo has been dug out from earlier days for showcasing. The original data is not mine, and the showcased results in `exploration.ipynb` are from the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html#dataset-zoo-open-images-v6).

Project Organization
------------

    ├── LICENSE            
    ├── README.md          <- Toplevel README
    ├── data
    │   ├── external       <- Testset and example-testset.
    │   ├── interim        <- Selected data.
    │   ├── processed      <- Images resized s.t. the largest dimension is 100.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models.
    │ 
    ├── architectures.py   <- Architectures implemented in PyTorch.
    ├── datasets.py        <- Dataset classes.
    ├── functional.py      <- Useful functions used in the project.
    ├── image_resizer.py   <- Script used to resize images.
    ├── filter_dataset.py  <- Script to filter data/raw to data/interrim
    ├── config.json        <- Network- and training-configuration.
    ├── train.py           <- Training script.

--------

## Remarks for grading

To reproduce results, run `generate_predictions.py`, with the final model being `models/final.pt` (only works if `data/external/challenge_testset.pkl`) is available.

