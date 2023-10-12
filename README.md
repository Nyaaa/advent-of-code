# 🎄Advent of Code🎄
Advent of Code puzzle solutions.

The solutions are mostly general and should work with all inputs, with some exceptions where data is hardcoded.

## Structure
Input data for each puzzle is stored in a private submodule, as per the [creator's request](https://twitter.com/ericwastl/status/1465805354214830081):
> In general I ask people not to publish their inputs, just to make it harder for someone to try to steal the whole
site.

The solutions rely on a helper library ['tools'](https://github.com/Nyaaa/advent-of-code/tree/master/tools), which contains some common functions, as well as parsers for the input files.
Parsers expect a text file, as given in the puzzle, with no modifications.
All input data is parsed, with very rare exceptions where parsing is not practical.


## Setup

This project uses [Conda](https://conda.io/projects/conda/en/latest/index.html) to manage virtual environments and
dependencies.

To create a new virtual environment and install all required libraries, run:\
`conda env create -f environment.yml`

Linting is handled by a global installation of Ruff, to enable pre-commit hooks run\
`pre-commit install`

## Completion

[![Completion Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Nyaaa/advent-of-code/master/2015/badge.json)](https://github.com/Nyaaa/advent-of-code/tree/master/2015)\
[![Completion Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Nyaaa/advent-of-code/master/2016/badge.json)](https://github.com/Nyaaa/advent-of-code/tree/master/2016)\
[![Completion Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Nyaaa/advent-of-code/master/2017/badge.json)](https://github.com/Nyaaa/advent-of-code/tree/master/2017)\
[![Completion Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Nyaaa/advent-of-code/master/2018/badge.json)](https://github.com/Nyaaa/advent-of-code/tree/master/2018)\
[![Completion Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Nyaaa/advent-of-code/master/2019/badge.json)](https://github.com/Nyaaa/advent-of-code/tree/master/2019)\
[![Completion Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Nyaaa/advent-of-code/master/2020/badge.json)](https://github.com/Nyaaa/advent-of-code/tree/master/2020)\
[![Completion Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Nyaaa/advent-of-code/master/2021/badge.json)](https://github.com/Nyaaa/advent-of-code/tree/master/2021)\
[![Completion Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Nyaaa/advent-of-code/master/2022/badge.json)](https://github.com/Nyaaa/advent-of-code/tree/master/2022)
