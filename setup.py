"""
lipyds
A toolkit for leaflet-based membrane analysis
"""
import sys
import os
import re
import shutil
import tempfile
import configparser
import platform
from subprocess import getoutput
from setuptools import setup, find_packages, Extension
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
import versioneer

RELEASE = "0.0.0-dev0"

is_release = 'dev' not in RELEASE

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

# set up cython extensions
try:
    import Cython
    from Cython.Build import cythonize
except ImportError:
    cython_found = False
    if not is_release:
        print("Lipyds requires Cython to compile extensions")
        sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Lipyds requires numpy")


def get_extensions():
    compile_args = ['-std=c++11', '-ffast-math', '-funroll-loops',
                    '-fsigned-zeros', '-O3']
    mathlib = [] if os.name == "nt" else ["m"]  # don't link maths for Windows

    # EXTENSION LIST

    # cutils = Extension(name="lipyds.lib.cutils",
    #                    sources=["lipyds/lib/cutils.pyx"],
    #                    include_dirs=[np.get_include()],
    #                    libraries=mathlib,
    #                    language="c++",
    #                    extra_compile_args=compile_args)

    cutils = Extension(name="lipyds.lib.cutils",
                       sources=["lipyds/lib/cutils.pyx"],
                       include_dirs=[np.get_include(), "lipyds/lib/include"],
                       libraries=mathlib,
                       language="c++",
                       extra_compile_args=compile_args)

    # COMPILATION LIST
    exts = [cutils]

    compiler_directives = {
        "embedsignature": False,
        "language_level": "3",
    }
    extensions = cythonize(exts, compiler_directives=compiler_directives)
    return extensions


if __name__ == "__main__":
    exts = get_extensions()

    try:
        with open("README.md", "r") as handle:
            long_description = handle.read()
    except:
        long_description = "\n".join(short_description[2:])

    install_requires = [
        'cython',
        'numpy>=1.16.0',
        'mdanalysis>=1.0.0',
        'mdanalysistests>=1.0.0',
    ]

    setup(
        # Self-descriptive entries which should always be present
        name='lipyds',
        author='Lily Wang',
        author_email='lily.wang@anu.edu.au',
        description=short_description[0],
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        license='MIT',

        # Which Python importable modules should be included when your package is installed
        # Handled automatically by setuptools. Use 'exclude' to prevent some specific
        # subpackage(s) from being added, if needed
        packages=find_packages(),

        # Optional include package data to ship with your package
        # Customize MANIFEST.in if the general case does not suit your needs
        # Comment out this line to prevent the files from being packaged with your software
        # include_package_data=True,
        ext_modules=exts,

        # Allows `setup.py test` to work correctly with pytest
        setup_requires=["numpy>=1.16.0"] + pytest_runner,

        # Additional entries you may want simply uncomment the lines you want and fill in the data
        # url='http://www.my_package.com',  # Website
        install_requires=install_requires,  # Required packages, pulls from pip if needed; do not use for Conda deployment
        platforms=['Linux',
                   'Mac OS-X',
                   'Unix',
                   'Windows'],            # Valid platforms your code works on, adjust to your flavor
        python_requires=">=3.6",          # Python version restrictions

        # Manual control if final package is compressible or not, set False to prevent the .egg from being made
        # zip_safe=False,

    )
