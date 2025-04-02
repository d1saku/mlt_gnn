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

## Graph Neural Network Implementation
Our custom GNN design:
- Learns embeddings of chromophores from graph representations
- Uses Avalon-descriptor of solvents to predict physical properties
- Represents molecules as graphs (atoms as nodes, bonds as edges)
- Utilizes DeepChem's MolGraphConv featurizer for node and edge descriptors including:
  - Atom charge, degree, and hybridization
  - Bond type, stereo configuration
  - Ring and conjugation information

## Comparison to classical ML models
Our research involved training classical ML models on molecular data (KNN, Tree-based boosting models, SVMs). 
As a result of comparative analysis, GNN was shown to be the best model overall for multi-task property prediction.
