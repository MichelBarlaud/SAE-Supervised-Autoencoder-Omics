# Pytorch implementation of the bilevel-projection l_{1,\infty}

This projection code is made useing the C++ extension of pytorch.

To install this package, please run the following command:
```
python setup.py install --user
```

Then, to test if the installation work, run the test.py file.
```
python test.py
```

The two function defined are the L1 ball projection:
```python
from l1inftyB_cpp import projL1
```

And the bilevel l_{1,\infty} projection:
```python
from l1inftyB_cpp import l1infty_bilevel
```