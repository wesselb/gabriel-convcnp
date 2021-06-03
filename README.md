# ConvCNP for Regression and Classification

An implementation of using the ConvCNP for regression and classification

## Installation

Clone and enter the repo.

```bash
git clone https://github.com/wesselb/gabriel-convcnp
cd gabriel-convcnp
```

Make and activate a virtual environment.

```bash
virtualenv -p python3.8 venv
source venv/bin/activate
```

Install an appropriate GPU-accelerated version of [PyTorch](https://pytorch.org/).

Finally, install the project.


```bash
pip install -e .
```

See also [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).

## Training the ConvCNP

Run

```bash
python train.py
```

For more information, see

```bash
python train.py --help
```

By default, results will be produced in `_experiments/experiment`, but you can change that by setting `--root some/other/path`.

Some details:

* `--alpha` controls the weight (value between `0` and `1`) assigned to the classification loss. By default, it is `0.5`.

*
    The data is generated from a sample of a GP on `[-2, 2]` with inputs uniformly sampled from this interval.

*
    The context set size is a random number between 3 and 50 and the target set size is always 50.
    
*
    The context and target sets are randomly split into classification and regression sets, and the classification targets are `1` if the value of the GP is above zero and `0` otherwise.

