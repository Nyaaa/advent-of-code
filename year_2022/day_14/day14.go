package main

import (
	"aoc/tools"
	"fmt"
	"regexp"
)

type cave struct {
	grid   [][]bool
	blocks int
	done   bool
}

func (c *cave) increaseDepth(rows int) {
	for i := 0; i < rows; i++ {
		newRow := make([]bool, len(c.grid[0]))
		c.grid = append(c.grid, newRow)
	}
}

func (c *cave) fall(row, column int) {
	switch {
	case c.grid[0][column] || row == len(c.grid):
		c.done = true
	case !c.grid[row][column]:
		c.fall(row+1, column)
	case !c.grid[row][column-1]:
		c.fall(row, column-1)
	case !c.grid[row][column+1]:
		c.fall(row, column+1)
	default:
		c.grid[row-1][column] = true
		c.blocks++
	}
}

func solve(input string, part2 bool) int {
	data := tools.ReadLines(input)
	nums := regexp.MustCompile(`\d+`)
	row := make([]bool, 700)
	cavern := cave{[][]bool{row}, 0, false}

	for _, line := range data {
		values := tools.StrToInt(nums.FindAllString(line, -1))

		for i := 0; i < len(values)-2; i += 2 {
			wallX, wallY := values[i], values[i+1]
			nextX, nextY := values[i+2], values[i+3]
			minX, maxX := min(wallX, nextX), max(wallX, nextX)
			minY, maxY := min(wallY, nextY), max(wallY, nextY)

			if maxY > len(cavern.grid) {
				add := maxY + 1 - len(cavern.grid)
				cavern.increaseDepth(add)
			}

			for i := minX; i <= maxX; i++ {
				cavern.grid[wallY][i] = true
			}

			for i := minY; i <= maxY; i++ {
				cavern.grid[i][wallX] = true
			}
		}
	}

	if part2 {
		cavern.increaseDepth(2)

		for i := 0; i < len(cavern.grid[0]); i++ {
			cavern.grid[len(cavern.grid)-1][i] = true
		}
	}

	for !cavern.done {
		cavern.fall(0, 500)
	}

	return cavern.blocks
}

func main() {
	testData := "test.txt"
	data := tools.GetData(2022, 14)

	fmt.Println(solve(testData, false), solve(testData, true))
	fmt.Println(solve(data, false), solve(data, true))
}
