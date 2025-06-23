# Fluorescent Molecule Property Prediction with Graph Neural Networks

## Project Overview

This project leverages Graph Neural Networks (GNNs) to predict the photophysical properties of fluorescent molecules. By learning from molecular structures, the models aim to accelerate the discovery and design of novel materials for applications like OLEDs, bioimaging, and chemical sensors.

The primary goal is to perform multi-task regression, predicting the following key properties from a molecule's chromophore and solvent:
-   Absorption Maximum (nm)
-   Emission Maximum (nm)
-   Fluorescence Quantum Yield
-   Fluorescence Lifetime (ns)

The core of the project is a hybrid model that processes chromophores as molecular graphs and solvents using their chemical fingerprints.

## Features

-   **GNN-based Models**: Implements several GNN architectures, including GCN, GIN, and GAT, located in [`models/model.py`](models/model.py).
-   **Hybrid Featurization**: Uses `deepchem` to create graph representations for chromophores and RDKit's Avalon fingerprints for solvents. See the featurization process in [`dataset/preprocess.py`](dataset/preprocess.py).
-   **Multi-Task Learning**: Predicts four distinct molecular properties simultaneously.
-   **Dynamic Loss Function**: Employs a custom loss function, [`LossWithMemory`](utils/utilities.py), which dynamically weights the loss for each target during training based on performance.
-   **Configurable Pipeline**: All training parameters, paths, and model settings are managed through a central configuration file at [`configs/default_config.yaml`](configs/default_config.yaml).
-   **Training and Evaluation**: Includes a streamlined script [`scripts/train.py`](scripts/train.py) for training a model from scratch and evaluating its performance using Mean Squared Error (MSE) and R² metrics.

## Project Structure

```
.
├── configs/              # Configuration files (e.g., default_config.yaml)
├── dataset/              # Data loading and preprocessing scripts
├── models/               # GNN model definitions
├── notebooks/            # Jupyter notebooks for experimentation and analysis
├── saved_models/         # Saved model weights and training history
├── scripts/              # Main scripts for training and evaluation
└── utils/                # Utility functions for loss, metrics, etc.
```

## Setup and Installation

1.  **Clone the repository**
2.  **Create a virtual environment** (recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies**: You will need libraries like PyTorch, PyTorch Geometric, RDKit, DeepChem, and scikit-learn.
    ```bash
    pip install torch torchvision torchaudio
    pip install torch-geometric
    pip install deepchem rdkit-pypi scikit-learn pyyaml
    ```

## Usage

### 1. Configuration

Adjust the training and model parameters in [`configs/default_config.yaml`](configs/default_config.yaml). You can modify the learning rate, batch size, number of epochs, and data paths.

### 2. Data

Place your dataset (e.g., `prep2.csv`) in the path specified by `dataset_path` in the config file (default is `dataset/processed`). The dataset should be a CSV file containing SMILES strings for the 'Chromophore' and 'Solvent', along with columns for the target properties.

### 3. Training

To start the training process, run the main training script. The script will automatically handle data splitting, preprocessing, training, and validation.

```bash
python scripts/train.py
```

The best model weights will be saved to the path specified by `model_save_path` in the config, and the loss history will be stored in a corresponding `.json` file.

### 4. Evaluation

The `test_model` function in [`scripts/train.py`](scripts/train.py) is called after the main script execution. It loads the best saved model and evaluates it on the test set, printing the final MSE and R² scores for each target. To run only the evaluation, you can comment out the `mlt_graph_training()` call in the script.