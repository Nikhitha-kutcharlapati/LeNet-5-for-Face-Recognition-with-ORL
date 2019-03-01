# LeNet-5 for Face Recognition with ORL

This implements a slightly modified LeNet-5 [LeCun et al., 1998a] and achieves an accuracy of ~98.75% on the [ORL dataset](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).

## Environment

- Anaconda 3 with Python 3.7.1

- Cuda 8.0

- PyTorch 1.0.0

- TorchVision 0.2.1

- Visdom 0.1.8.8

- MATLAB 2017

## Usage

Run the `data.m` to randomly generate `trainlist.txt` and `testlist.txt`

```
$ matlab -nojvm -r data.m
```

Start the `visdom` server for visualization

```
$ python -m visdom.server
```

Start the training procedure

```
$ python run.py
```

See epoch train loss live graph at [`http://localhost:8097`](http://localhost:8097).

## References

[[1](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.

[[2](https://github.com/activatedgeek/LeNet-5)] PyTorch implementation of LeNet-5 with live visualization.
