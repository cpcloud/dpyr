from setuptools import setup, find_packages

import versioneer


with open('README.md', 'rt') as f:
    long_description = f.read()


with open('requirements.txt', 'rt') as f:
    install_requires = [line.strip() for line in f]


all_requires = [
    'ibis-framework[impala, kerberos, pandas, postgres, sqlite, visualization]'
]
develop_requires = all_requires + ['flake8', 'pytest >= 3', 'mypy-lang']


setup(
    name='dpyr',
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    extras_require={
        'all': all_requires,
        'develop': develop_requires,
        'impala': ['ibis-framework[impala, kerberos]'],
        'pandas': ['ibis-framework[pandas]'],
        'postgres': ['ibis-framework[postgres]'],
        'sqlite': ['ibis-framework[sqlite]'],
        'visualization': ['ibis-framework[visualization]'],
    },
    description=(
        'Python dplyr operations for SQL databases and pandas DataFrames'
    ),
    long_description=long_description,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    license='MIT',
    maintainer='Phillip Cloud',
)
