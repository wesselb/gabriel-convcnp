# ConvCNP for Regression and Classification

An implementation of using the ConvCNP for regression and classification

## Installation

Clone and enter the repo.

```bash
git clone https://github.com/wesselb/convcnp
cd convcnp
```

Finally, make and activate a virtual environment.

```bash
virtualenv -p python3 venv
```

Install an appropriate GPU-accelerated version of [PyTorch](https://pytorch.org/).
Finally, install the project.


```bash
source venv/bin/activate
pip install -e .
```

See also [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).

## Training the ConvCNP

To get started, run

```bash
python train.py --help
```

By default, results will be produced in `_experiments/experiment`.

