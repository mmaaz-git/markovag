# README
Exact sensitivity analysis of Markov reward processes via algebraic geometry

## Getting Started with markovag

`markovag` is a Python package for performing sensitivity analyses in cost-effectiveness analysis with algebraic geometry. The basic idea is that all such analyses can be reformulated as systems of polynomial inequalities. We can then construct a cylindrical algebraic decomposition (CAD) of this system in order to determine the entire parameter space over which our objective holds.

The folder `markovag` contains the `markovag` Python package. To use it, you can either navigate to the folder and install it via `pip install .` or place the folder at the same level as your code files. The package contains two modules: `markovag.markov` which allows for doing cost-effectiveness analyses symbolically, and `markovag.cad` which actually constructs the CAD for your sensitivity analysis. You can learn how to use both modules by reading the Jupyter notebooks `markov.ipynb` and `cad.ipynb`, which each contain numerous examples and explanation provided.

## Reproducing the case studies

Our synthetic case study from the paper is contained in `synthetic.ipynb`. The case study that re-analyzes the cost-effectiveness of drones is contained in `drones.ipynb`. This uses several data files in `drones_files`, which contains a table of patient data and tables of costs and utilities. The patient data is a "noisified" version of the original patient data used in the paper. We cannot share the original one due to data privacy concerns. Hence, the analysis in `drones.ipynb` is not exactly the same as in the paper, but is close.



