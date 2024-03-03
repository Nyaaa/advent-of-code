package main

import (
	"aoc/tools"
	"fmt"
	"regexp"
)

type cave [][]int

func increaseDepth(cavern *cave, rows int) {
	for i := 0; i < rows; i++ {
		newRow := make([]int, len((*cavern)[0]))
		*cavern = append(*cavern, newRow)
	}
}

func solve(input string) int {
	nums := regexp.MustCompile(`\d+`)
	data := tools.ReadLines(input)
	left, right := 500, 700
	cavern := cave{make([]int, right)}

	for _, line := range data {
		nn := nums.FindAllString(line, -1)
		values := tools.StrToInt(nn)

		for i := 0; i < len(values)-2; i += 2 {
			wallX, wallY := values[i], values[i+1]
			nextX, nextY := values[i+2], values[i+3]
			maxX, minX := max(wallX, nextX), min(wallX, nextX)
			maxY, minY := max(wallY, nextY), min(wallY, nextY)
			left = min(minX, left)
			right = max(maxX, right)

			if maxY > len(cavern) {
				add := maxY + 1 - len(cavern)
				increaseDepth(&cavern, add)
			}

			for i := minX; i <= maxX; i++ {
				cavern[wallY][i] = 1
			}

			for i := minY; i <= maxY; i++ {
				cavern[i][wallX] = 1
			}
		}
	}

	result := 0

	return result
}

func main() {
	testData := "test.txt"
	// data := tools.GetData(2022, 14)

	fmt.Println(solve(testData))
}
