# MolEncoder
Molecular AutoEncoder in PyTorch

## Install

```bash
$ git clone https://github.com/cxhernandez/molencoder.git && cd molencoder
$ python setup.py install
```

## Download Dataset

```bash
$ molencoder download --dataset chembl22
```

## Train

```bash
$ molencoder train --dataset data/chembl22.h5
```

Add `--cuda` flag to enable CUDA. Add `--cont` to continue training a model from a checkpoint file.


## Pre-Trained Model

A pre-trained reference model is available in the `ref/` directory. Currently, it performs with ~98% accuracy on the validation set after 100 epochs of training. However, if you succeed at training a better model, feel free to submit a pull request!

## TODO

- [x] Implement encoder
- [x] Implement decoder
- [x] Add download command
- [x] Add train command
- [ ] Add encode command
- [ ] Add decode command
- [x] Add pre-trained model


## Shoutouts

+ [Original paper](https://arxiv.org/abs/1610.02415) by Gómez-Bombarelli, et al.
+ [keras-molecules](https://github.com/maxhodak/keras-molecules) by Max Hodak
+ [DeepChem](https://github.com/deepchem/deepchem)
+ [PyTorch](pytorch.org)
