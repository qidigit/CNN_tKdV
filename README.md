# An MS-D Net to capture extreme events

## Problem description

This repository shows an example of using machine learning strategy to predict extreme events in complex systems: A densely connected multi-scale network model (MS-D Net) is applied to capture the extreme events appearing in a truncated Korteweg–de Vries (tKdV) statistical framework which generates transition from near-Gaussian statistics to anomalous skewed distributions consistent with recent laboratory experiments for shallow water waves across an abrupt depth change.

## To run an experiment

The `main.py` script is used to run the experiment. 
To train the neural network model without using a pretrained checkpoint, run the following command:

```
python main.py --exp_dir=<EXP_DIR> --cfg=<CONFIG_PATH> --nopretrained --write_data --train
```

To test the trained model with the path to the latest checkpoint, run the following command:

```
python main.py --exp_dir=<EXP_DIR> --cfg=<CONFIG_PATH> --pretrained --write_data --notrain
```

## Dataset

Datasets for training and prediction in the neural network model are generated from direct numerical simulations of the tKdV equation in different statistical regimes:

* training dataset 'tKdV_J32th10': model with truncation size $J=32$ and inverse temperature $\theta = -0.1$, showing near-Gaussian statistics in solutions;
* prediction dataset 'tKdV_J32th50': model with truncation size $J=32$ and inverse temperature $\theta = -0.5$, showing highly skewed statistics in solutions.

Different datasets can be tested by changing the configuration file, config.py. A wider variety of problems in different statistical regimes can be also tested by adding new corresponding dataset into the data/ folder.

## Dependencies

* [PyTorch >= 1.2.0](https://pytorch.org)

## References
[1] D. Qi and A. J. Majda, “Using machine learning to predict extreme events in complex systems,” Proceedings of the National Academy of Sciences, 2019.  <br />
[2] A. J. Majda, M. Moore, and D. Qi, “Statistical dynamical model to predict extreme events and anomalous features in shallow water waves with abrupt depth change,” Proceedings of the National Academy of Sciences, [vol. 116, no. 10, pp. 3982–3987](https://www.pnas.org/content/116/10/3982), 2019.
