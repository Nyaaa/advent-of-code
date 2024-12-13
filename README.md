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

This project uses [UV](https://docs.astral.sh/uv/) to manage virtual environments and
dependencies.

To create a new virtual environment and install all required libraries, run:\
`uv sync`

Linting is handled by a global installation of Ruff, to enable pre-commit hooks run\
`pre-commit install`

<!-- AOC TILES BEGIN -->
<h1 align="center">
  Advent of Code - 476/476 ⭐
</h1>
<h1 align="center">
  2024 - 26 ⭐ - Python
</h1>
<a href="year_2024/day_01/day01.py">
  <img src=".aoc_tiles/tiles/2024/01.png" width="161px">
</a>
<a href="year_2024/day_02/day02.py">
  <img src=".aoc_tiles/tiles/2024/02.png" width="161px">
</a>
<a href="year_2024/day_03/day03.py">
  <img src=".aoc_tiles/tiles/2024/03.png" width="161px">
</a>
<a href="year_2024/day_04/day04.py">
  <img src=".aoc_tiles/tiles/2024/04.png" width="161px">
</a>
<a href="year_2024/day_05/day05.py">
  <img src=".aoc_tiles/tiles/2024/05.png" width="161px">
</a>
<a href="year_2024/day_06/day06.py">
  <img src=".aoc_tiles/tiles/2024/06.png" width="161px">
</a>
<a href="year_2024/day_07/day07.py">
  <img src=".aoc_tiles/tiles/2024/07.png" width="161px">
</a>
<a href="year_2024/day_08/day08.py">
  <img src=".aoc_tiles/tiles/2024/08.png" width="161px">
</a>
<a href="year_2024/day_09/day09.py">
  <img src=".aoc_tiles/tiles/2024/09.png" width="161px">
</a>
<a href="year_2024/day_10/day10.py">
  <img src=".aoc_tiles/tiles/2024/10.png" width="161px">
</a>
<a href="year_2024/day_11/day11.py">
  <img src=".aoc_tiles/tiles/2024/11.png" width="161px">
</a>
<a href="year_2024/day_12/day12.py">
  <img src=".aoc_tiles/tiles/2024/12.png" width="161px">
</a>
<a href="year_2024/day_13/day13.py">
  <img src=".aoc_tiles/tiles/2024/13.png" width="161px">
</a>
<h1 align="center">
  2023 - 50 ⭐ - Python
</h1>
<a href="year_2023/day_01/day01.go">
  <img src=".aoc_tiles/tiles/2023/01.png" width="161px">
</a>
<a href="year_2023/day_02/day02.go">
  <img src=".aoc_tiles/tiles/2023/02.png" width="161px">
</a>
<a href="year_2023/day_03/day03.go">
  <img src=".aoc_tiles/tiles/2023/03.png" width="161px">
</a>
<a href="year_2023/day_04/day04.go">
  <img src=".aoc_tiles/tiles/2023/04.png" width="161px">
</a>
<a href="year_2023/day_05/day05.go">
  <img src=".aoc_tiles/tiles/2023/05.png" width="161px">
</a>
<a href="year_2023/day_06/day06.go">
  <img src=".aoc_tiles/tiles/2023/06.png" width="161px">
</a>
<a href="year_2023/day_07/day07.go">
  <img src=".aoc_tiles/tiles/2023/07.png" width="161px">
</a>
<a href="year_2023/day_08/day08.go">
  <img src=".aoc_tiles/tiles/2023/08.png" width="161px">
</a>
<a href="year_2023/day_09/day09.go">
  <img src=".aoc_tiles/tiles/2023/09.png" width="161px">
</a>
<a href="year_2023/day_10/day10.go">
  <img src=".aoc_tiles/tiles/2023/10.png" width="161px">
</a>
<a href="year_2023/day_11/day11.go">
  <img src=".aoc_tiles/tiles/2023/11.png" width="161px">
</a>
<a href="year_2023/day_12/day12.py">
  <img src=".aoc_tiles/tiles/2023/12.png" width="161px">
</a>
<a href="year_2023/day_13/day13.py">
  <img src=".aoc_tiles/tiles/2023/13.png" width="161px">
</a>
<a href="year_2023/day_14/day14.py">
  <img src=".aoc_tiles/tiles/2023/14.png" width="161px">
</a>
<a href="year_2023/day_15/day15.py">
  <img src=".aoc_tiles/tiles/2023/15.png" width="161px">
</a>
<a href="year_2023/day_16/day16.py">
  <img src=".aoc_tiles/tiles/2023/16.png" width="161px">
</a>
<a href="year_2023/day_17/day17.py">
  <img src=".aoc_tiles/tiles/2023/17.png" width="161px">
</a>
<a href="year_2023/day_18/day18.py">
  <img src=".aoc_tiles/tiles/2023/18.png" width="161px">
</a>
<a href="year_2023/day_19/day19.py">
  <img src=".aoc_tiles/tiles/2023/19.png" width="161px">
</a>
<a href="year_2023/day_20/day20.py">
  <img src=".aoc_tiles/tiles/2023/20.png" width="161px">
</a>
<a href="year_2023/day_21/day21.py">
  <img src=".aoc_tiles/tiles/2023/21.png" width="161px">
</a>
<a href="year_2023/day_22/day22.py">
  <img src=".aoc_tiles/tiles/2023/22.png" width="161px">
</a>
<a href="year_2023/day_23/day23.py">
  <img src=".aoc_tiles/tiles/2023/23.png" width="161px">
</a>
<a href="year_2023/day_24/day24.py">
  <img src=".aoc_tiles/tiles/2023/24.png" width="161px">
</a>
<a href="year_2023/day_25/day25.py">
  <img src=".aoc_tiles/tiles/2023/25.png" width="161px">
</a>
<h1 align="center">
  2022 - 50 ⭐ - Python
</h1>
<a href="year_2022/day_01/day01.go">
  <img src=".aoc_tiles/tiles/2022/01.png" width="161px">
</a>
<a href="year_2022/day_02/day02.go">
  <img src=".aoc_tiles/tiles/2022/02.png" width="161px">
</a>
<a href="year_2022/day_03/day03.go">
  <img src=".aoc_tiles/tiles/2022/03.png" width="161px">
</a>
<a href="year_2022/day_04/day04.go">
  <img src=".aoc_tiles/tiles/2022/04.png" width="161px">
</a>
<a href="year_2022/day_05/day05.go">
  <img src=".aoc_tiles/tiles/2022/05.png" width="161px">
</a>
<a href="year_2022/day_06/day06.go">
  <img src=".aoc_tiles/tiles/2022/06.png" width="161px">
</a>
<a href="year_2022/day_07/day07.go">
  <img src=".aoc_tiles/tiles/2022/07.png" width="161px">
</a>
<a href="year_2022/day_08/day08.go">
  <img src=".aoc_tiles/tiles/2022/08.png" width="161px">
</a>
<a href="year_2022/day_09/day09.go">
  <img src=".aoc_tiles/tiles/2022/09.png" width="161px">
</a>
<a href="year_2022/day_10/day10.go">
  <img src=".aoc_tiles/tiles/2022/10.png" width="161px">
</a>
<a href="year_2022/day_11/day11.go">
  <img src=".aoc_tiles/tiles/2022/11.png" width="161px">
</a>
<a href="year_2022/day_12/day12.go">
  <img src=".aoc_tiles/tiles/2022/12.png" width="161px">
</a>
<a href="year_2022/day_13/day13.go">
  <img src=".aoc_tiles/tiles/2022/13.png" width="161px">
</a>
<a href="year_2022/day_14/day14.go">
  <img src=".aoc_tiles/tiles/2022/14.png" width="161px">
</a>
<a href="year_2022/day_15/day15.py">
  <img src=".aoc_tiles/tiles/2022/15.png" width="161px">
</a>
<a href="year_2022/day_16/day16.py">
  <img src=".aoc_tiles/tiles/2022/16.png" width="161px">
</a>
<a href="year_2022/day_17/day17.py">
  <img src=".aoc_tiles/tiles/2022/17.png" width="161px">
</a>
<a href="year_2022/day_18/day18.py">
  <img src=".aoc_tiles/tiles/2022/18.png" width="161px">
</a>
<a href="year_2022/day_19/day19.py">
  <img src=".aoc_tiles/tiles/2022/19.png" width="161px">
</a>
<a href="year_2022/day_20/day20.py">
  <img src=".aoc_tiles/tiles/2022/20.png" width="161px">
</a>
<a href="year_2022/day_21/day21.py">
  <img src=".aoc_tiles/tiles/2022/21.png" width="161px">
</a>
<a href="year_2022/day_22/day22.py">
  <img src=".aoc_tiles/tiles/2022/22.png" width="161px">
</a>
<a href="year_2022/day_23/day23.py">
  <img src=".aoc_tiles/tiles/2022/23.png" width="161px">
</a>
<a href="year_2022/day_24/day24.py">
  <img src=".aoc_tiles/tiles/2022/24.png" width="161px">
</a>
<a href="year_2022/day_25/day25.py">
  <img src=".aoc_tiles/tiles/2022/25.png" width="161px">
</a>
<h1 align="center">
  2021 - 50 ⭐ - Python
</h1>
<a href="year_2021/day_01/day01.py">
  <img src=".aoc_tiles/tiles/2021/01.png" width="161px">
</a>
<a href="year_2021/day_02/day02.py">
  <img src=".aoc_tiles/tiles/2021/02.png" width="161px">
</a>
<a href="year_2021/day_03/day03.py">
  <img src=".aoc_tiles/tiles/2021/03.png" width="161px">
</a>
<a href="year_2021/day_04/day04.py">
  <img src=".aoc_tiles/tiles/2021/04.png" width="161px">
</a>
<a href="year_2021/day_05/day05.py">
  <img src=".aoc_tiles/tiles/2021/05.png" width="161px">
</a>
<a href="year_2021/day_06/day06.py">
  <img src=".aoc_tiles/tiles/2021/06.png" width="161px">
</a>
<a href="year_2021/day_07/day07.py">
  <img src=".aoc_tiles/tiles/2021/07.png" width="161px">
</a>
<a href="year_2021/day_08/day08.py">
  <img src=".aoc_tiles/tiles/2021/08.png" width="161px">
</a>
<a href="year_2021/day_09/day09.py">
  <img src=".aoc_tiles/tiles/2021/09.png" width="161px">
</a>
<a href="year_2021/day_10/day10.py">
  <img src=".aoc_tiles/tiles/2021/10.png" width="161px">
</a>
<a href="year_2021/day_11/day11.py">
  <img src=".aoc_tiles/tiles/2021/11.png" width="161px">
</a>
<a href="year_2021/day_12/day12.py">
  <img src=".aoc_tiles/tiles/2021/12.png" width="161px">
</a>
<a href="year_2021/day_13/day13.py">
  <img src=".aoc_tiles/tiles/2021/13.png" width="161px">
</a>
<a href="year_2021/day_14/day14.py">
  <img src=".aoc_tiles/tiles/2021/14.png" width="161px">
</a>
<a href="year_2021/day_15/day15.py">
  <img src=".aoc_tiles/tiles/2021/15.png" width="161px">
</a>
<a href="year_2021/day_16/day16.py">
  <img src=".aoc_tiles/tiles/2021/16.png" width="161px">
</a>
<a href="year_2021/day_17/day17.py">
  <img src=".aoc_tiles/tiles/2021/17.png" width="161px">
</a>
<a href="year_2021/day_18/day18.py">
  <img src=".aoc_tiles/tiles/2021/18.png" width="161px">
</a>
<a href="year_2021/day_19/day19.py">
  <img src=".aoc_tiles/tiles/2021/19.png" width="161px">
</a>
<a href="year_2021/day_20/day20.py">
  <img src=".aoc_tiles/tiles/2021/20.png" width="161px">
</a>
<a href="year_2021/day_21/day21.py">
  <img src=".aoc_tiles/tiles/2021/21.png" width="161px">
</a>
<a href="year_2021/day_22/day22.py">
  <img src=".aoc_tiles/tiles/2021/22.png" width="161px">
</a>
<a href="year_2021/day_23/day23.py">
  <img src=".aoc_tiles/tiles/2021/23.png" width="161px">
</a>
<a href="year_2021/day_24/day24.py">
  <img src=".aoc_tiles/tiles/2021/24.png" width="161px">
</a>
<a href="year_2021/day_25/day25.py">
  <img src=".aoc_tiles/tiles/2021/25.png" width="161px">
</a>
<h1 align="center">
  2020 - 50 ⭐ - Python
</h1>
<a href="year_2020/day_01/day01.py">
  <img src=".aoc_tiles/tiles/2020/01.png" width="161px">
</a>
<a href="year_2020/day_02/day02.py">
  <img src=".aoc_tiles/tiles/2020/02.png" width="161px">
</a>
<a href="year_2020/day_03/day03.py">
  <img src=".aoc_tiles/tiles/2020/03.png" width="161px">
</a>
<a href="year_2020/day_04/day04.py">
  <img src=".aoc_tiles/tiles/2020/04.png" width="161px">
</a>
<a href="year_2020/day_05/day05.py">
  <img src=".aoc_tiles/tiles/2020/05.png" width="161px">
</a>
<a href="year_2020/day_06/day06.py">
  <img src=".aoc_tiles/tiles/2020/06.png" width="161px">
</a>
<a href="year_2020/day_07/day07.py">
  <img src=".aoc_tiles/tiles/2020/07.png" width="161px">
</a>
<a href="year_2020/day_08/day08.py">
  <img src=".aoc_tiles/tiles/2020/08.png" width="161px">
</a>
<a href="year_2020/day_09/day09.py">
  <img src=".aoc_tiles/tiles/2020/09.png" width="161px">
</a>
<a href="year_2020/day_10/day10.py">
  <img src=".aoc_tiles/tiles/2020/10.png" width="161px">
</a>
<a href="year_2020/day_11/day11.py">
  <img src=".aoc_tiles/tiles/2020/11.png" width="161px">
</a>
<a href="year_2020/day_12/day12.py">
  <img src=".aoc_tiles/tiles/2020/12.png" width="161px">
</a>
<a href="year_2020/day_13/day13.py">
  <img src=".aoc_tiles/tiles/2020/13.png" width="161px">
</a>
<a href="year_2020/day_14/day14.py">
  <img src=".aoc_tiles/tiles/2020/14.png" width="161px">
</a>
<a href="year_2020/day_15/day15.py">
  <img src=".aoc_tiles/tiles/2020/15.png" width="161px">
</a>
<a href="year_2020/day_16/day16.py">
  <img src=".aoc_tiles/tiles/2020/16.png" width="161px">
</a>
<a href="year_2020/day_17/day17.py">
  <img src=".aoc_tiles/tiles/2020/17.png" width="161px">
</a>
<a href="year_2020/day_18/day18.py">
  <img src=".aoc_tiles/tiles/2020/18.png" width="161px">
</a>
<a href="year_2020/day_19/day19.py">
  <img src=".aoc_tiles/tiles/2020/19.png" width="161px">
</a>
<a href="year_2020/day_20/day20.py">
  <img src=".aoc_tiles/tiles/2020/20.png" width="161px">
</a>
<a href="year_2020/day_21/day21.py">
  <img src=".aoc_tiles/tiles/2020/21.png" width="161px">
</a>
<a href="year_2020/day_22/day22.py">
  <img src=".aoc_tiles/tiles/2020/22.png" width="161px">
</a>
<a href="year_2020/day_23/day23.py">
  <img src=".aoc_tiles/tiles/2020/23.png" width="161px">
</a>
<a href="year_2020/day_24/day24.py">
  <img src=".aoc_tiles/tiles/2020/24.png" width="161px">
</a>
<a href="year_2020/day_25/day25.py">
  <img src=".aoc_tiles/tiles/2020/25.png" width="161px">
</a>
<h1 align="center">
  2019 - 50 ⭐ - Python
</h1>
<a href="year_2019/day_01/day01.py">
  <img src=".aoc_tiles/tiles/2019/01.png" width="161px">
</a>
<a href="year_2019/day_02/day02.py">
  <img src=".aoc_tiles/tiles/2019/02.png" width="161px">
</a>
<a href="year_2019/day_03/day03.py">
  <img src=".aoc_tiles/tiles/2019/03.png" width="161px">
</a>
<a href="year_2019/day_04/day04.py">
  <img src=".aoc_tiles/tiles/2019/04.png" width="161px">
</a>
<a href="year_2019/day_05/day05.py">
  <img src=".aoc_tiles/tiles/2019/05.png" width="161px">
</a>
<a href="year_2019/day_06/day06.py">
  <img src=".aoc_tiles/tiles/2019/06.png" width="161px">
</a>
<a href="year_2019/day_07/day07.py">
  <img src=".aoc_tiles/tiles/2019/07.png" width="161px">
</a>
<a href="year_2019/day_08/day08.py">
  <img src=".aoc_tiles/tiles/2019/08.png" width="161px">
</a>
<a href="year_2019/day_09/day09.py">
  <img src=".aoc_tiles/tiles/2019/09.png" width="161px">
</a>
<a href="year_2019/day_10/day10.py">
  <img src=".aoc_tiles/tiles/2019/10.png" width="161px">
</a>
<a href="year_2019/day_11/day11.py">
  <img src=".aoc_tiles/tiles/2019/11.png" width="161px">
</a>
<a href="year_2019/day_12/day12.py">
  <img src=".aoc_tiles/tiles/2019/12.png" width="161px">
</a>
<a href="year_2019/day_13/day13.py">
  <img src=".aoc_tiles/tiles/2019/13.png" width="161px">
</a>
<a href="year_2019/day_14/day14.py">
  <img src=".aoc_tiles/tiles/2019/14.png" width="161px">
</a>
<a href="year_2019/day_15/day15.py">
  <img src=".aoc_tiles/tiles/2019/15.png" width="161px">
</a>
<a href="year_2019/day_16/day16.py">
  <img src=".aoc_tiles/tiles/2019/16.png" width="161px">
</a>
<a href="year_2019/day_17/day17.py">
  <img src=".aoc_tiles/tiles/2019/17.png" width="161px">
</a>
<a href="year_2019/day_18/day18.py">
  <img src=".aoc_tiles/tiles/2019/18.png" width="161px">
</a>
<a href="year_2019/day_19/day19.py">
  <img src=".aoc_tiles/tiles/2019/19.png" width="161px">
</a>
<a href="year_2019/day_20/day20.py">
  <img src=".aoc_tiles/tiles/2019/20.png" width="161px">
</a>
<a href="year_2019/day_21/day21.py">
  <img src=".aoc_tiles/tiles/2019/21.png" width="161px">
</a>
<a href="year_2019/day_22/day22.py">
  <img src=".aoc_tiles/tiles/2019/22.png" width="161px">
</a>
<a href="year_2019/day_23/day23.py">
  <img src=".aoc_tiles/tiles/2019/23.png" width="161px">
</a>
<a href="year_2019/day_24/day24.py">
  <img src=".aoc_tiles/tiles/2019/24.png" width="161px">
</a>
<a href="year_2019/day_25/day25.py">
  <img src=".aoc_tiles/tiles/2019/25.png" width="161px">
</a>
<h1 align="center">
  2018 - 50 ⭐ - Python
</h1>
<a href="year_2018/day_01/day01.py">
  <img src=".aoc_tiles/tiles/2018/01.png" width="161px">
</a>
<a href="year_2018/day_02/day02.py">
  <img src=".aoc_tiles/tiles/2018/02.png" width="161px">
</a>
<a href="year_2018/day_03/day03.py">
  <img src=".aoc_tiles/tiles/2018/03.png" width="161px">
</a>
<a href="year_2018/day_04/day04.py">
  <img src=".aoc_tiles/tiles/2018/04.png" width="161px">
</a>
<a href="year_2018/day_05/day05.py">
  <img src=".aoc_tiles/tiles/2018/05.png" width="161px">
</a>
<a href="year_2018/day_06/day06.py">
  <img src=".aoc_tiles/tiles/2018/06.png" width="161px">
</a>
<a href="year_2018/day_07/day07.py">
  <img src=".aoc_tiles/tiles/2018/07.png" width="161px">
</a>
<a href="year_2018/day_08/day08.py">
  <img src=".aoc_tiles/tiles/2018/08.png" width="161px">
</a>
<a href="year_2018/day_09/day09.py">
  <img src=".aoc_tiles/tiles/2018/09.png" width="161px">
</a>
<a href="year_2018/day_10/day10.py">
  <img src=".aoc_tiles/tiles/2018/10.png" width="161px">
</a>
<a href="year_2018/day_11/day11.py">
  <img src=".aoc_tiles/tiles/2018/11.png" width="161px">
</a>
<a href="year_2018/day_12/day12.py">
  <img src=".aoc_tiles/tiles/2018/12.png" width="161px">
</a>
<a href="year_2018/day_13/day13.py">
  <img src=".aoc_tiles/tiles/2018/13.png" width="161px">
</a>
<a href="year_2018/day_14/day14.py">
  <img src=".aoc_tiles/tiles/2018/14.png" width="161px">
</a>
<a href="year_2018/day_15/day15.py">
  <img src=".aoc_tiles/tiles/2018/15.png" width="161px">
</a>
<a href="year_2018/day_16/day16.py">
  <img src=".aoc_tiles/tiles/2018/16.png" width="161px">
</a>
<a href="year_2018/day_17/day17.py">
  <img src=".aoc_tiles/tiles/2018/17.png" width="161px">
</a>
<a href="year_2018/day_18/day18.py">
  <img src=".aoc_tiles/tiles/2018/18.png" width="161px">
</a>
<a href="year_2018/day_19/day19.py">
  <img src=".aoc_tiles/tiles/2018/19.png" width="161px">
</a>
<a href="year_2018/day_20/day20.py">
  <img src=".aoc_tiles/tiles/2018/20.png" width="161px">
</a>
<a href="year_2018/day_21/day21.py">
  <img src=".aoc_tiles/tiles/2018/21.png" width="161px">
</a>
<a href="year_2018/day_22/day22.py">
  <img src=".aoc_tiles/tiles/2018/22.png" width="161px">
</a>
<a href="year_2018/day_23/day23.py">
  <img src=".aoc_tiles/tiles/2018/23.png" width="161px">
</a>
<a href="year_2018/day_24/day24.py">
  <img src=".aoc_tiles/tiles/2018/24.png" width="161px">
</a>
<a href="year_2018/day_25/day25.py">
  <img src=".aoc_tiles/tiles/2018/25.png" width="161px">
</a>
<h1 align="center">
  2017 - 50 ⭐ - Python
</h1>
<a href="year_2017/day_01/day01.py">
  <img src=".aoc_tiles/tiles/2017/01.png" width="161px">
</a>
<a href="year_2017/day_02/day02.py">
  <img src=".aoc_tiles/tiles/2017/02.png" width="161px">
</a>
<a href="year_2017/day_03/day03.py">
  <img src=".aoc_tiles/tiles/2017/03.png" width="161px">
</a>
<a href="year_2017/day_04/day04.py">
  <img src=".aoc_tiles/tiles/2017/04.png" width="161px">
</a>
<a href="year_2017/day_05/day05.py">
  <img src=".aoc_tiles/tiles/2017/05.png" width="161px">
</a>
<a href="year_2017/day_06/day06.py">
  <img src=".aoc_tiles/tiles/2017/06.png" width="161px">
</a>
<a href="year_2017/day_07/day07.py">
  <img src=".aoc_tiles/tiles/2017/07.png" width="161px">
</a>
<a href="year_2017/day_08/day08.py">
  <img src=".aoc_tiles/tiles/2017/08.png" width="161px">
</a>
<a href="year_2017/day_09/day09.py">
  <img src=".aoc_tiles/tiles/2017/09.png" width="161px">
</a>
<a href="year_2017/day_10/day10.py">
  <img src=".aoc_tiles/tiles/2017/10.png" width="161px">
</a>
<a href="year_2017/day_11/day11.py">
  <img src=".aoc_tiles/tiles/2017/11.png" width="161px">
</a>
<a href="year_2017/day_12/day12.py">
  <img src=".aoc_tiles/tiles/2017/12.png" width="161px">
</a>
<a href="year_2017/day_13/day13.py">
  <img src=".aoc_tiles/tiles/2017/13.png" width="161px">
</a>
<a href="year_2017/day_14/day14.py">
  <img src=".aoc_tiles/tiles/2017/14.png" width="161px">
</a>
<a href="year_2017/day_15/day15.py">
  <img src=".aoc_tiles/tiles/2017/15.png" width="161px">
</a>
<a href="year_2017/day_16/day16.py">
  <img src=".aoc_tiles/tiles/2017/16.png" width="161px">
</a>
<a href="year_2017/day_17/day17.py">
  <img src=".aoc_tiles/tiles/2017/17.png" width="161px">
</a>
<a href="year_2017/day_18/day18.py">
  <img src=".aoc_tiles/tiles/2017/18.png" width="161px">
</a>
<a href="year_2017/day_19/day19.py">
  <img src=".aoc_tiles/tiles/2017/19.png" width="161px">
</a>
<a href="year_2017/day_20/day20.py">
  <img src=".aoc_tiles/tiles/2017/20.png" width="161px">
</a>
<a href="year_2017/day_21/day21.py">
  <img src=".aoc_tiles/tiles/2017/21.png" width="161px">
</a>
<a href="year_2017/day_22/day22.py">
  <img src=".aoc_tiles/tiles/2017/22.png" width="161px">
</a>
<a href="year_2017/day_23/day23.py">
  <img src=".aoc_tiles/tiles/2017/23.png" width="161px">
</a>
<a href="year_2017/day_24/day24.py">
  <img src=".aoc_tiles/tiles/2017/24.png" width="161px">
</a>
<a href="year_2017/day_25/day25.py">
  <img src=".aoc_tiles/tiles/2017/25.png" width="161px">
</a>
<h1 align="center">
  2016 - 50 ⭐ - Python
</h1>
<a href="year_2016/day_01/day01.py">
  <img src=".aoc_tiles/tiles/2016/01.png" width="161px">
</a>
<a href="year_2016/day_02/day02.py">
  <img src=".aoc_tiles/tiles/2016/02.png" width="161px">
</a>
<a href="year_2016/day_03/day03.py">
  <img src=".aoc_tiles/tiles/2016/03.png" width="161px">
</a>
<a href="year_2016/day_04/day04.py">
  <img src=".aoc_tiles/tiles/2016/04.png" width="161px">
</a>
<a href="year_2016/day_05/day05.py">
  <img src=".aoc_tiles/tiles/2016/05.png" width="161px">
</a>
<a href="year_2016/day_06/day06.py">
  <img src=".aoc_tiles/tiles/2016/06.png" width="161px">
</a>
<a href="year_2016/day_07/day07.py">
  <img src=".aoc_tiles/tiles/2016/07.png" width="161px">
</a>
<a href="year_2016/day_08/day08.py">
  <img src=".aoc_tiles/tiles/2016/08.png" width="161px">
</a>
<a href="year_2016/day_09/day09.py">
  <img src=".aoc_tiles/tiles/2016/09.png" width="161px">
</a>
<a href="year_2016/day_10/day10.py">
  <img src=".aoc_tiles/tiles/2016/10.png" width="161px">
</a>
<a href="year_2016/day_11/day11.py">
  <img src=".aoc_tiles/tiles/2016/11.png" width="161px">
</a>
<a href="year_2016/day_12/day12.py">
  <img src=".aoc_tiles/tiles/2016/12.png" width="161px">
</a>
<a href="year_2016/day_13/day13.py">
  <img src=".aoc_tiles/tiles/2016/13.png" width="161px">
</a>
<a href="year_2016/day_14/day14.py">
  <img src=".aoc_tiles/tiles/2016/14.png" width="161px">
</a>
<a href="year_2016/day_15/day15.py">
  <img src=".aoc_tiles/tiles/2016/15.png" width="161px">
</a>
<a href="year_2016/day_16/day16.py">
  <img src=".aoc_tiles/tiles/2016/16.png" width="161px">
</a>
<a href="year_2016/day_17/day17.py">
  <img src=".aoc_tiles/tiles/2016/17.png" width="161px">
</a>
<a href="year_2016/day_18/day18.py">
  <img src=".aoc_tiles/tiles/2016/18.png" width="161px">
</a>
<a href="year_2016/day_19/day19.py">
  <img src=".aoc_tiles/tiles/2016/19.png" width="161px">
</a>
<a href="year_2016/day_20/day20.py">
  <img src=".aoc_tiles/tiles/2016/20.png" width="161px">
</a>
<a href="year_2016/day_21/day21.py">
  <img src=".aoc_tiles/tiles/2016/21.png" width="161px">
</a>
<a href="year_2016/day_22/day22.py">
  <img src=".aoc_tiles/tiles/2016/22.png" width="161px">
</a>
<a href="year_2016/day_23/day23.py">
  <img src=".aoc_tiles/tiles/2016/23.png" width="161px">
</a>
<a href="year_2016/day_24/day24.py">
  <img src=".aoc_tiles/tiles/2016/24.png" width="161px">
</a>
<a href="year_2016/day_25/day25.py">
  <img src=".aoc_tiles/tiles/2016/25.png" width="161px">
</a>
<h1 align="center">
  2015 - 50 ⭐ - Python
</h1>
<a href="year_2015/day_01/day01.ipynb">
  <img src=".aoc_tiles/tiles/2015/01.png" width="161px">
</a>
<a href="year_2015/day_02/day02.ipynb">
  <img src=".aoc_tiles/tiles/2015/02.png" width="161px">
</a>
<a href="year_2015/day_03/day03.ipynb">
  <img src=".aoc_tiles/tiles/2015/03.png" width="161px">
</a>
<a href="year_2015/day_04/day04.ipynb">
  <img src=".aoc_tiles/tiles/2015/04.png" width="161px">
</a>
<a href="year_2015/day_05/day05.ipynb">
  <img src=".aoc_tiles/tiles/2015/05.png" width="161px">
</a>
<a href="year_2015/day_06/day06.ipynb">
  <img src=".aoc_tiles/tiles/2015/06.png" width="161px">
</a>
<a href="year_2015/day_07/day07.ipynb">
  <img src=".aoc_tiles/tiles/2015/07.png" width="161px">
</a>
<a href="year_2015/day_08/day08.ipynb">
  <img src=".aoc_tiles/tiles/2015/08.png" width="161px">
</a>
<a href="year_2015/day_09/day09.ipynb">
  <img src=".aoc_tiles/tiles/2015/09.png" width="161px">
</a>
<a href="year_2015/day_10/day10.ipynb">
  <img src=".aoc_tiles/tiles/2015/10.png" width="161px">
</a>
<a href="year_2015/day_11/day11.ipynb">
  <img src=".aoc_tiles/tiles/2015/11.png" width="161px">
</a>
<a href="year_2015/day_12/day12.ipynb">
  <img src=".aoc_tiles/tiles/2015/12.png" width="161px">
</a>
<a href="year_2015/day_13/day13.ipynb">
  <img src=".aoc_tiles/tiles/2015/13.png" width="161px">
</a>
<a href="year_2015/day_14/day14.ipynb">
  <img src=".aoc_tiles/tiles/2015/14.png" width="161px">
</a>
<a href="year_2015/day_15/day15.ipynb">
  <img src=".aoc_tiles/tiles/2015/15.png" width="161px">
</a>
<a href="year_2015/day_16/day16.ipynb">
  <img src=".aoc_tiles/tiles/2015/16.png" width="161px">
</a>
<a href="year_2015/day_17/day17.ipynb">
  <img src=".aoc_tiles/tiles/2015/17.png" width="161px">
</a>
<a href="year_2015/day_18/day18.ipynb">
  <img src=".aoc_tiles/tiles/2015/18.png" width="161px">
</a>
<a href="year_2015/day_19/day19.py">
  <img src=".aoc_tiles/tiles/2015/19.png" width="161px">
</a>
<a href="year_2015/day_20/day20.py">
  <img src=".aoc_tiles/tiles/2015/20.png" width="161px">
</a>
<a href="year_2015/day_21/day21.ipynb">
  <img src=".aoc_tiles/tiles/2015/21.png" width="161px">
</a>
<a href="year_2015/day_22/day22.ipynb">
  <img src=".aoc_tiles/tiles/2015/22.png" width="161px">
</a>
<a href="year_2015/day_23/day23.ipynb">
  <img src=".aoc_tiles/tiles/2015/23.png" width="161px">
</a>
<a href="year_2015/day_24/day24.py">
  <img src=".aoc_tiles/tiles/2015/24.png" width="161px">
</a>
<a href="year_2015/day_25/day25.py">
  <img src=".aoc_tiles/tiles/2015/25.png" width="161px">
</a>
<!-- AOC TILES END -->
