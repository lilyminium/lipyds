lipyds
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/lilyminium/lipyds/workflows/CI/badge.svg)](https://github.com/lilyminium/lipyds/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/lilyminium/lipyds/branch/master/graph/badge.svg)](https://codecov.io/gh/lilyminium/lipyds/branch/master)


A toolkit for leaflet-based membrane analysis


## Installation

A release will be forthcoming. Until then, you can build this from source.

```python
git clone https://github.com/lilyminium/lipyds.git
conda create --name lipyds -c conda-forge cython matplotlib nptyping "numpy>=1.20.0" pandas "scikit-learn>=0.21.0" scipy
conda activate lipyds
cd lipyds
python setup.py install
```

### Copyright

Copyright (c) 2021, Lily Wang
