package main

import (
	"fmt"
	"github.com/Nyaaa/advent-of-code/tools"
	"gonum.org/v1/gonum/stat/combin"
)

func getGalaxyMap(data []string) ([][2]int, map[int]bool, map[int]bool) {
	rows, cols := map[int]bool{}, map[int]bool{}
	galaxies := [][2]int{}

	for rowIndex, row := range data {
		lineNums := []int{}
		zeroes := 0

		for colIndex, char := range row {
			if char == '#' {
				lineNums = append(lineNums, 1)
				galaxies = append(galaxies, [2]int{rowIndex, colIndex})
			} else {
				lineNums = append(lineNums, 0)
				zeroes++
			}
		}

		if len(lineNums) == zeroes {
			rows[rowIndex] = true
		}
	}

	for i := 0; i < len(data[0]); i++ {
		zeroes := 0

		for j := 0; j < len(data); j++ {
			if data[j][i] != '#' {
				zeroes++
			}
		}

		if zeroes == len(data) {
			cols[i] = true
		}
	}

	return galaxies, rows, cols
}

func solve(input string, multiplier int) int {
	distance := 0
	galaxies, rows, cols := getGalaxyMap(tools.ReadLines(input))

	for _, combo := range combin.Combinations(len(galaxies), 2) {
		galaxyA, galaxyB := galaxies[combo[0]], galaxies[combo[1]]
		emptyRows, emptyCols := 0, 0
		minRow, maxRow := tools.MinMax(galaxyA[0], galaxyB[0])
		minCol, maxCol := tools.MinMax(galaxyA[1], galaxyB[1])

		for i := minRow; i < maxRow; i++ {
			if rows[i] {
				emptyRows++
			}
		}

		for i := minCol; i < maxCol; i++ {
			if cols[i] {
				emptyCols++
			}
		}

		distance += tools.Abs(galaxyA[0]-galaxyB[0]) + emptyRows*(multiplier-1)
		distance += tools.Abs(galaxyA[1]-galaxyB[1]) + emptyCols*(multiplier-1)
	}

	return distance
}

func main() {
	fmt.Println(solve("test.txt", 2))
	fmt.Println(solve("test.txt", 100))
	fmt.Println(solve(tools.GetData(2023, 11), 2))
	fmt.Println(solve(tools.GetData(2023, 11), 1_000_000))
}
