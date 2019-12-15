# A MS-D Net to capture extreme events

## Problem

This repository shows a simple example of using machine learning to predict extreme events in complex systems: a densely connected mixed-scale network model is used to capture the extreme events appearing in a truncated Korteweg–de Vries (tKdV) statistical framework which creates anomalous skewed distributions consistent with recent laboratory experiments for shallow water waves across an abrupt depth change

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

## Dependencies

* [PyTorch >= 1.2.0](https://pytorch.org)

## References
[1] D. Qi and A. J. Majda, “Using machine learning to predict extreme events in complex systems,” Proceedings of the National Academy of Sciences, 2019.  <br />
[2] A. J. Majda, M. Moore, and D. Qi, “Statistical dynamical model to predict extreme events and anomalous features in shallow water waves with abrupt depth change,” Proceedings of the National Academy of Sciences, [vol. 116, no. 10, pp. 3982–3987](https://www.pnas.org/content/116/10/3982), 2019.
