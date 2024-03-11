package main

import (
	"aoc/tools"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"unicode"
)

func getAdjacent(row, col, limit int) [][2]int {
	adjacent := [][2]int{}

	for r := -1; r <= 1; r++ {
		for c := -1; c <= 1; c++ {
			adj := [2]int{row + r, col + c}
			if adj[0] >= 0 && adj[1] >= 0 && adj[0] < limit && adj[1] < limit {
				adjacent = append(adjacent, adj)
			}
		}
	}

	return adjacent
}

func solvePart2(gears map[[2]int][]int) int {
	part2 := 0

	for _, values := range gears {
		slices.Sort(values)
		c := slices.Compact(values)

		if len(c) > 1 {
			ratio := 1
			for _, num := range c {
				ratio *= num
			}

			part2 += ratio
		}
	}

	return part2
}

func solve(input string) (int, int) {
	part1 := 0
	nums := regexp.MustCompile(`\d+`)
	grid := tools.ReadLines(input)
	gears := map[[2]int][]int{}

	for i, row := range grid {
		for _, part := range nums.FindAllStringIndex(row, -1) {
			partValue := 0

			for j := part[0]; j < part[1]; j++ {
				for _, adjIndex := range getAdjacent(i, j, len(grid)) {
					adj := grid[adjIndex[0]][adjIndex[1]]
					if !unicode.IsNumber(rune(adj)) && adj != '.' {
						partValue, _ = strconv.Atoi(row[part[0]:part[1]])

						if adj == '*' {
							gears[adjIndex] = append(gears[adjIndex], partValue)
						}
					}
				}
			}

			if partValue != 0 {
				part1 += partValue
			}
		}
	}

	return part1, solvePart2(gears)
}

func main() {
	fmt.Println(solve("test.txt"))
	fmt.Println(solve(tools.GetData(2023, 3)))
}
