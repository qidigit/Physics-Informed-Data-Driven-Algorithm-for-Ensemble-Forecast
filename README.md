# Physics-Informed-Data-Driven-Algorithm-for-Ensemble-Forecast
Physics-Informed Data-Driven Algorithm (PIDD-CG) for Ensemble Forecast of Complex Turbulent Systems

## Problem description

This repository implements the PIDD-CG algorithm described in [1] for predictnig the PDFs in complex turbulent systems. The model reduction method employs a Long-Short-Term-Memory architecture to represent the higher-order unresolved statistical feedbacks with careful consideration to account for the conditional Gaussian structure yet producing highly non-Gaussian statistics. 

## To run an experiment

Three models are provides to run the experiment under different truncation scenarios:

`train_dyad_condGau.py` and `pred_dyad_condGau.py`: training and prediction for the dyad model

`train_baroflow_condGau.py` and `pred_baroflow_condGau.py`: training and prediction for the high dimensional barotropic model


To train the neural network model without using a pretrained checkpoint, run the following command:

```
python train_*_condGau.py --exp_dir=<EXP_DIR> --pretrained FALSE --eval FALSE
```

To test the trained model with the path to the latest checkpoint, run the following command:

```
python train_*_condGau.py --exp_dir=<EXP_DIR> --pretrained TRUE --eval TRUE
```

## Dataset

Datasets for training and prediction in the neural network model are generated from direct Monte-Carlo simulations of the L-96 system:

* datasets 'dyad_su05sv2' and 'baro_K10sk20sU10dk1dU1': model statistics for training and prediction in a long time trajectory

A wider variety of problems in different perturbation scenarios can be also tested by adding new corresponding dataset into the data/ folder.

## Dependencies

* [PyTorch >= 1.2.0](https://pytorch.org)

## References
[1] N. Chen and D. Qi  (2022), “A Physics-Informed Data-Driven Algorithm for Ensemble Forecast of Complex Turbulent Systems,” arXiv:.
