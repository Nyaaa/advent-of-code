package main

import (
	"aoc/tools"
	"fmt"
	"regexp"
)

type cave [][]int

func (c *cave) increaseDepth(rows int) {
	for i := 0; i < rows; i++ {
		newRow := make([]int, len((*c)[0]))
		*c = append(*c, newRow)
	}
}

func drawCave(data []string, part2 bool) (cave, int, int) {
	nums := regexp.MustCompile(`\d+`)
	left, right := 500, 700
	cavern := cave{make([]int, right)}

	for _, line := range data {
		values := tools.StrToInt(nums.FindAllString(line, -1))

		for i := 0; i < len(values)-2; i += 2 {
			wallX, wallY := values[i], values[i+1]
			nextX, nextY := values[i+2], values[i+3]
			minX, maxX := min(wallX, nextX), max(wallX, nextX)
			minY, maxY := min(wallY, nextY), max(wallY, nextY)
			left, right = min(minX, left), max(maxX, right)

			if maxY > len(cavern) {
				add := maxY + 1 - len(cavern)
				cavern.increaseDepth(add)
			}

			for i := minX; i <= maxX; i++ {
				cavern[wallY][i] = 1
			}

			for i := minY; i <= maxY; i++ {
				cavern[i][wallX] = 1
			}
		}
	}

	if part2 {
		cavern.increaseDepth(2)

		for i := 0; i < len(cavern[0]); i++ {
			cavern[len(cavern)-1][i] = 1
		}
	}

	return cavern, left, right
}

func fall(cavern *cave, left, right *int, row, column int) bool {
	c := (*cavern)
	result := true

	switch {
	case c[0][column] == 1 || row == len(c):
		return false
	case c[row][column] != 1:
		result = fall(cavern, left, right, row+1, column)
	case c[row][column-1] != 1:
		result = fall(cavern, left, right, row, column-1)
	case c[row][column+1] != 1:
		result = fall(cavern, left, right, row, column+1)
	default:
		c[row-1][column] = 1
		*left = min(*left, column)
		*right = max(*right, column)
	}

	return result
}

func solve(input string, part2 bool) int {
	cavern, left, right := drawCave(tools.ReadLines(input), part2)
	counter := 0
	placed := true

	for placed {
		placed = fall(&cavern, &left, &right, 0, 500)

		if placed {
			counter++
		}
	}

	return counter
}

func main() {
	testData := "test.txt"
	data := tools.GetData(2022, 14)

	fmt.Println(solve(testData, false), solve(testData, true))
	fmt.Println(solve(data, false), solve(data, true))
}
