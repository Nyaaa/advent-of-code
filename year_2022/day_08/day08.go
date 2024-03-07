package main

import (
	"aoc/tools"
	"fmt"
	"slices"
	"strings"
)

func solve(input string) (int, int) {
	var (
		forest       = [][]int{}
		data         = tools.ReadLines(input)
		part1, part2 int
	)

	for _, line := range data {
		if line == "" {
			continue
		}

		line := strings.Split(line, "")
		forest = append(forest, tools.StrToInt(line))
	}

	for row := 0; row < len(forest); row++ {
		for col := 0; col < len(forest); col++ {
			visible, score := evaluateTile(row, col, forest)
			part2 = max(part2, score)
			part1 += visible
		}
	}

	return part1, part2
}

func evaluateTile(row int, col int, forest [][]int) (int, int) {
	left := forest[row][:col]
	left = append(left[:0:0], left...)
	right := forest[row][col+1:]
	top := []int{}
	bottom := []int{}

	for i := 0; i < len(forest); i++ {
		tile := forest[i][col]
		if i < row {
			top = append(top, tile)
		} else if i > row {
			bottom = append(bottom, tile)
		}
	}

	slices.Reverse(top)
	slices.Reverse(left)

	return getScore([][]int{left, right, top, bottom}, forest[row][col])
}

func getScore(vision [][]int, tile int) (int, int) {
	isVisible := 0
	score := 1

	for _, line := range vision {
		canSee := 0
		clear := true

		for _, tree := range line {
			canSee++

			if tree >= tile {
				clear = false

				break
			}
		}

		score *= canSee

		if clear {
			isVisible = 1
		}
	}

	return isVisible, score
}

func main() {
	fmt.Println(solve("test.txt"))
	fmt.Println(solve(tools.GetData(2022, 8)))
}
