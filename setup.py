from setuptools import setup, find_packages

setup(
    name='markovag',
    version='0.1',
    author="Muhammad Maaz",
    author_email="m.maaz@mail.utoronto.ca",
    description='A Python package for cost-effectiveness analysis with Markov reward processes and cylindrical algebraic decomposition.',
    packages=find_packages(),
    install_requires=[
        'anytree==2.12.1',
        'sympy==1.12.1'
    ],
    python_requires='>=3.10'
)