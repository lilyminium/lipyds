lipyds
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/lilyminium/lipyds/workflows/CI/badge.svg)](https://github.com/lilyminium/lipyds/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/lilyminium/lipyds/branch/master/graph/badge.svg)](https://codecov.io/gh/lilyminium/lipyds/branch/master)
[![Documentation Status](https://readthedocs.org/projects/lipyds/badge/?version=latest)](https://lipyds.readthedocs.io/en/latest/?badge=latest)
[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)


A toolkit for leaflet-based membrane analysis

**Note: this code was used for a publication requiring membrane analysis.
It is still in early development status and should be regarded as experimental --
the API is very likely to change in the future. Please use at your own risk!**

**Note: this package is temporarily stripped back to just supporting the LeafletFinder
to unblock core MDAnalysis development.**

The initial code was first used in the below publication. Please cite it if
you use this package:

Wilson, K. A.; Wang, L.; Lin, Y. C.; O’Mara, M. L.
Investigating the Lipid Fingerprint of SLC6 Neurotransmitter Transporters:
A Comparison of DDAT, HDAT, HSERT, and GlyT2. BBA Advances 2021, 1, 100010.
doi: [10.1016/j.bbadva.2021.100010](https://doi.org/10.1016/j.bbadva.2021.100010)

The package is built on MDAnalysis. Please cite its two papers if you use this
package:

N. Michaud-Agrawal, E. J. Denning, T. B. Woolf,
and O. Beckstein. MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics
Simulations. *J. Comput. Chem.* **32** (2011),
2319–2327. doi:10.1002/jcc.21787.

R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N.
Melo, S. L. Seyler, D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney,
and O. Beckstein. MDAnalysis: A Python package for the rapid analysis of
molecular dynamics simulations. In S. Benthall and S. Rostrup, editors,
*Proceedings of the 15th Python in Science Conference*, pages 98-105,
Austin, TX, 2016. SciPy. doi:10.25080/Majora-629e541a-00e.

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
