# SIR Model Inference: Stan vs Neural Network

This project compares Bayesian inference using Stan and deep learning inference using a neural network for parameter estimation in a stochastic SIR (Susceptible-Infected-Recovered) epidemiological model.

## Structure
- `evaluation.ipynb` — Main notebook for evaluating and comparing Stan and NN inference.  
- `nn_definition.ipynb` — Neural network architecture and training routines.  
- `global_utils.py` — Utilities for data generation, model evaluation, and helper functions.  
- `train_best_model.py` — Script to train and save the best neural network model.  
- `hyperparam_tuning.py` — Hyperparameter optimization using Optuna.  
- `sir-demo.stan` — Stan model definition for the SIR process.  
- `requirements.txt` — Python dependencies