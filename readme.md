# DSN20230220

## Description

Code associated with the Data Science Nashville Meetup for 2023-02-20: [New Tricks for Old Data Scientists: Scikit-Learn](https://www.meetup.com/data-science-nashville/events/291391952/). This is intended as a showcase for some new(-ish) features in Scikit-learn. Note that this code is intended to be instructional and as such should not be considered production-level/ready.

## Instructions

### Data

This repository uses data from the [NIH Chest X-rays dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data). Specifically, the [Random Sample of NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample) available on Kaggle. This is a subset of the full 112K dataset consisting of 5606 images of size 1024x1024. The data can also be downloaded directly from the NIH: [ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345).

Extract images into the data directory as follows:

```
DSN20230220
├───data
│   └───nih
│       └───images
├───notebooks
├───output
└───scripts
```

### Environment

The conda environment is specified by sklearn1dot2.yml. Install with the following command:

`conda env create -f sklearn1dot2.yml`


## Contact

Jason King (jason@kinglambda.com)
