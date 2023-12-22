# Multivariate-time-series-domain-adaptation
(Still working on modifying codes for clarity.)

Implementations of some domain adaptation algorithms with simple codes in Pytorch.

Some codes are originally made for image DA, but I modified for multi-variate time-series DA.

## Algorithms Implemented
| File name    | Description                                    |
| ---------- | ---------------------------------------------- |
| 1. Bidirectional | [Bidirectional One-Shot Unsupervised Domain Mapping](https://arxiv.org/abs/1909.01595)|
| 2. (recurrent DANN) R-DANN   | [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)|
| 3. CoDATS   | [Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data](https://arxiv.org/abs/1801.01290) |
| 3. VRADA   | [Variational Recurrent Adversarial Deep Domain Adaptation](https://openreview.net/pdf?id=rk9eAFcxg) |
| 4. DSN   | [Domain Separation Networks](https://arxiv.org/abs/1801.01290) |

If you want R-DANN, CoDATS, VRADA codes, see "DANN.py" file. These codes are based on DANN, so only the backbone networks are different.
You can simple train these algorithm by changing the backbone in this file.

## Dependencies
Pytorch
Numpy


