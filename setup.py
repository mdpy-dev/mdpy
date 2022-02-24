from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description  =  fh.read()

setup(
    name = 'mdpy',
    version = '0.2.1',
    author = 'Zhenyu Wei',
    author_email = 'zhenyuwei99@gmail.com',
    description = 'MDPy is a python framework for conducting highly extensible and flexible molecular dynamics simulation',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords = 'Peptide dynamics simulation package',
    classifiers  =  [
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9'
    ],
    url = 'https://mdpy.net/',
    project_urls = {
        "Documentation": "https://mdpy.net/",
        "Source Code": "https://github.com/mdpy-dev/mdpy",
    },
    packages = find_packages(),
    package_data = {
        "mdpy": [
            "test/data/*", 
        ]
    },
    # setup_requires = ['pytest-runner'],
    tests_require = ['pytest', 'pytest-xdist'],
    install_requires = [
        'numpy >= 1.20.0',
        'scipy >= 1.7.0',
        'numba >= 0.54.0',
        'matplotlib >= 3.0.0',
        'pytest >= 6.2.0',
        'pytest-xdist >= 2.3.0',
        'MDAnalysis >= 2.0.0',
        'h5py >= 3.6.0'
    ],
    python_requires = '>=3.9'
)
