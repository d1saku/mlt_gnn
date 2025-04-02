# Fluorescent Molecule Prediction and Generation

## Project Overview
This project focuses on the development of machine learning models for the prediction and generation of fluorescent molecules, which are essential components in various applications including organic light-emitting diodes (OLEDs), bioimaging, chemical sensors, and dyes.

Traditional discovery and development of fluorescent molecules rely on trial-and-error methods and theoretical calculations using Density Functional Theory, which demands significant computational resources. Our project addresses these challenges by applying machine learning techniques to streamline the discovery process and property prediction.

### Property Prediction
We have developed regression models trained on a dataset of fluorescent molecules to predict key properties:
- Absorption maxima
- Emission maxima
- Quantum yield
- Lifetime

Our approach employs various molecular vector encodings:
- Morgan Fingerprints
- Avalon Fingerprints
- Molecular ACCess System Fingerprint

Prediction models include:
- Classical machine learning algorithms:
  - Linear regression
  - Lasso regression
  - KNN regression
  - Ridge regression
- Deep learning:
  - Graph Neural Network (GNN)

### Molecule Generation
We utilize two pre-trained models for generating novel molecular structures:
- Chemical Language Model (CLM)
- Transmol

## Pipeline Architecture
1. Dataset division into training, testing, and validation subsets
2. Training with various molecular representations and machine learning models
3. Randomized cross-validation for hyperparameter tuning
4. Selection of the three most effective model-representation pairs
5. Implementation of ensemble learning techniques (averaging outputs)
6. Property prediction for newly synthesized molecules

## Graph Neural Network Implementation
Our custom GNN design:
- Learns embeddings of chromophores from graph representations
- Uses Avalon-descriptor of solvents to predict physical properties
- Represents molecules as graphs (atoms as nodes, bonds as edges)
- Utilizes DeepChem's MolGraphConv featurizer for node and edge descriptors including:
  - Atom charge, degree, and hybridization
  - Bond type, stereo configuration
  - Ring and conjugation information
