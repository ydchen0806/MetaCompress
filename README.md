# Meta-Learning with MAML

This repository contains scripts for meta-learning and model compression. The scripts are designed to work with segmented data blocks.

## Environment Setup

To set up the required environment, you can choose from the following options:

- **Using pip**:
  You can install the necessary Python dependencies from the `requirements.txt` file using the following command:

  ```bash
  pip install -r requirements.txt

We highly recommend using Docker to set up the required environment. Two Docker images are available for your convenience:

- **Using Docker from ali cloud**:
  - [**registry.cn-hangzhou.aliyuncs.com/cyd_dl/monai-vit:v26**](https://registry.cn-hangzhou.aliyuncs.com/cyd_dl/monai-vit:v26)
  
  ```bash
  docker pull registry.cn-hangzhou.aliyuncs.com/cyd_dl/monai-vit:v26
  
- **Using docker from dockerhub**:
  - [**cyd_docker:v1**](https://ydchen0806/cyd_docker:v1)
  
  ```bash
  docker pull ydchen0806/cyd_docker:v1

## Scripts

### `MAML.py`

This script is used to perform meta-learning training on segmented data blocks.

### `train_from_meta.py`

This script starts the model compression process using the learned initialization parameters from meta-learning.

### `train_from_random.py`

This script begins the model compression process with randomly initialized parameters.

### `train_data.py`

This script is used to fit the entire original dataset starting from the learned initialization parameters from meta-learning.

### `train_data_random.py`

This script fits the entire original dataset starting from randomly initialized parameters.

## Configuration and File Locations

All configuration files and script files are located in the `/ssd/0/qjy/MAML/Run_pipeline/` directory. Configuration files are in YAML format, and a single configuration file can be used for both meta-learning training and data compression.

## Running the Scripts

- To perform meta-learning training and data compression on segmented data blocks, run `overfit.sh`.

- To complete meta-learning training and compression on the original data, run `train_data.sh`.

Make sure you have the necessary dependencies and datasets set up before running these scripts. Refer to the provided configuration files for further customization of the training and compression processes.
