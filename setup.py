"""
mdkit_leaflets
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
    cython_found = True
    from distutils.version import LooseVersion

    required_version = "0.16"
    if not LooseVersion(Cython.__version__) >= LooseVersion(required_version):
        # We don't necessarily die here. Maybe we already have
        #  the cythonized '.c' files.
        print("Cython version {0} was found but won't be used: version {1} "
              "or greater is required because it offers a handy "
              "parallelization module".format(
               Cython.__version__, required_version))
        cython_found = False
    cython_linetrace = bool(os.environ.get('CYTHON_TRACE_NOGIL', False))
except ImportError as e:
    cython_found = False
    if not is_release:
        print("*** package: Cython not found ***")
        print("MDKit Leaflets requires Cython for development builds")
        sys.exit(1)
    cython_linetrace = False


def get_numpy_include():
    # Obtain the numpy include directory. This logic works across numpy
    # versions.
    # setuptools forgets to unset numpy's setup flag and we get a crippled
    # version of it unless we do it ourselves.
    import builtins

    builtins.__NUMPY_SETUP__ = False
    try:
        import numpy as np
    except ImportError:
        print('*** package "numpy" not found ***')
        print('This code requires a version of NumPy (>=1.16.0), even for setup.')
        print('Please get it from http://numpy.scipy.org/ or install it through '
              'your package manager.')
        sys.exit(-1)
    return np.get_include()


def detect_openmp():
    """Does this compiler support OpenMP parallelization?"""
    print("Attempting to autodetect OpenMP support... ", end="")
    compiler = new_compiler()
    customize_compiler(compiler)
    compiler.add_library('gomp')
    include = '<omp.h>'
    extra_postargs = ['-fopenmp']
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads()', include=include,
                            extra_postargs=extra_postargs)
    if hasopenmp:
        print("Compiler supports OpenMP")
    else:
        print("Did not detect OpenMP support.")
    return hasopenmp


def hasfunction(cc, funcname, include=None, extra_postargs=None):
    # From http://stackoverflow.com/questions/
    #            7018879/disabling-output-when-compiling-with-distutils
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            with open(fname, 'w') as f:
                if include is not None:
                    f.write('#include {0!s}\n'.format(include))
                f.write('int main(void) {\n')
                f.write('    {0!s};\n'.format(funcname))
                f.write('}\n')
            # Redirect stderr to /dev/null to hide any error messages
            # from the compiler.
            # This will have to be changed if we ever have to check
            # for a function on Windows.
            devnull = open('/dev/null', 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            objects = cc.compile([fname], output_dir=tmpdir,
                                 extra_postargs=extra_postargs)
            cc.link_executable(objects, os.path.join(tmpdir, "a.out"))
        except Exception:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)

class Config(object):
    """Config wrapper class to get build options

    This class looks for options in the environment variables and the
    'setup.cfg' file. The order how we look for an option is.

    1. Environment Variable
    2. set in 'setup.cfg'
    3. given default

    Environment variables should start with 'MDA_' and be all uppercase.
    Values passed to environment variables are checked (case-insensitively)
    for specific strings with boolean meaning: 'True' or '1' will cause `True`
    to be returned. '0' or 'False' cause `False` to be returned.

    """

    def __init__(self, fname='setup.cfg'):
        fname = abspath(fname)
        if os.path.exists(fname):
            self.config = configparser.ConfigParser()
            self.config.read(fname)

    def get(self, option_name, default=None):
        environ_name = 'MDA_' + option_name.upper()
        if environ_name in os.environ:
            val = os.environ[environ_name]
            if val.upper() in ('1', 'TRUE'):
                return True
            elif val.upper() in ('0', 'FALSE'):
                return False
            return val
        try:
            option = self.config.get('options', option_name)
            return option
        except (configparser.NoOptionError, configparser.NoSectionError):
            return default

def abspath(file):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        file)
class MDAExtension(Extension, object):
    """Derived class to cleanly handle setup-time (numpy) dependencies.
    """
    # The only setup-time numpy dependency comes when setting up its
    #  include dir.
    # The actual numpy import and call can be delayed until after pip
    #  has figured it must install numpy.
    # This is accomplished by passing the get_numpy_include function
    #  as one of the include_dirs. This derived Extension class takes
    #  care of calling it when needed.
    
    def __init__(self, name, sources, *args, **kwargs):
        self._mda_include_dirs = []
        sources = [abspath(s) for s in sources]
        super(MDAExtension, self).__init__(name, sources, *args, **kwargs)

    @property
    def include_dirs(self):
        if not self._mda_include_dirs:
            for item in self._mda_include_dir_args:
                try:
                    self._mda_include_dirs.append(item()) #The numpy callable
                except TypeError:
                    item = abspath(item)
                    self._mda_include_dirs.append((item))
        return self._mda_include_dirs

    @include_dirs.setter
    def include_dirs(self, val):
        self._mda_include_dir_args = val


def using_clang():
    """Will we be using a clang compiler?"""
    compiler = new_compiler()
    customize_compiler(compiler)
    compiler_ver = getoutput("{0} -v".format(compiler.compiler[0]))
    return 'clang' in compiler_ver


def extensions(config):
    use_cython = config.get('use_cython', default=not is_release)
    use_openmp = config.get('use_openmp', default=True)

    extra_compile_args = ['-std=c++11', '-ffast-math', '-funroll-loops',
                          '-fsigned-zeros', '-O3']
    extra_link_args = []
    define_macros = []
    if config.get('debug_cflags', default=False):
        extra_compile_args.extend(['-Wall', '-pedantic'])
        define_macros.extend([('DEBUG', '1')])

    # allow using architecture specific instructions. This allows people to
    # build optimized versions of the code.
    arch = config.get('march', default=False)
    if arch:
        extra_compile_args.append('-march={}'.format(arch))

    if platform.system() == 'Darwin' and using_clang():
        extra_compile_args.append('-stdlib=libc++')
        extra_compile_args.append('-mmacosx-version-min=10.9')
        extra_link_args.append('-stdlib=libc++')
        extra_link_args.append('-mmacosx-version-min=10.7')
    
    has_openmp = detect_openmp()

    if use_openmp and not has_openmp:
        print('No openmp compatible compiler found default to serial build.')

    parallel_args = ['-fopenmp'] if has_openmp and use_openmp else []
    parallel_libraries = ['gomp'] if has_openmp and use_openmp else []
    parallel_macros = [('PARALLEL', None)] if has_openmp and use_openmp else []

    if use_cython:
        print('Will attempt to use Cython.')
        if not cython_found:
            print("Couldn't find a Cython installation. "
                  "Not recompiling cython extensions.")
            use_cython = False
    else:
        print('Will not attempt to use Cython.')

    cpp_source_suffix = '.pyx' if use_cython else '.cpp'

    include_dirs = [get_numpy_include]
    # Windows automatically handles math library linking
    # and will not build if we try to specify one
    if os.name == 'nt':
        mathlib = []
    else:
        mathlib = ['m']
    
    if cython_linetrace:
        extra_compile_args.append("-DCYTHON_TRACE_NOGIL")
        cpp_extra_compile_args.append("-DCYTHON_TRACE_NOGIL")
    
    cutils = MDAExtension("mdkleaflets.lib.cutils", 
                          ["mdkleaflets/lib/cutils" + cpp_source_suffix],
                          include_dirs=include_dirs,
                          libraries=mathlib,
                          language="c++",
                          define_macros=define_macros,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args)
    
    pre_exts = [cutils]
    cython_generated = []
    if use_cython:
        extensions = cythonize(
            pre_exts,
            compiler_directives={'linetrace': cython_linetrace,
                                 'embedsignature': False,
                                 'language_level': '3'},
        )
        if cython_linetrace:
            print("Cython coverage will be enabled")
        for pre_ext, post_ext in zip(pre_exts, extensions):
            for source in post_ext.sources:
                if source not in pre_ext.sources:
                    cython_generated.append(source)
    else:
        #Let's check early for missing .c files
        extensions = pre_exts
        for ext in extensions:
            for source in ext.sources:
                if not (os.path.isfile(source) and
                        os.access(source, os.R_OK)):
                    raise IOError("Source file '{}' not found. This might be "
                                "caused by a missing Cython install, or a "
                                "failed/disabled Cython build.".format(source))
    return extensions, cython_generated



if __name__ == "__main__":
    config = Config()
    exts, cythonfiles = extensions(config)

    try:
        with open("README.md", "r") as handle:
            long_description = handle.read()
    except:
        long_description = "\n".join(short_description[2:])

    install_requires = [
            'numpy>=1.16.0',
            'mdanalysis>=1.0.0',
            'mdanalysistests>=1.0.0',
        ]

    setup(
        # Self-descriptive entries which should always be present
        name='mdkit_leaflets',
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
        include_package_data=True,
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
        python_requires=">=3.5",          # Python version restrictions

        # Manual control if final package is compressible or not, set False to prevent the .egg from being made
        # zip_safe=False,

    )
