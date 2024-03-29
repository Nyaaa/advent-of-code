package main

import (
	"aoc/tools"
	"fmt"
	"slices"
	"strings"
)

func extrapolate(scan []int) int {
	layers := [][]int{scan}
	index := 0

	for {
		layer := layers[index]
		newLayer := []int{}
		zeroes := 0

		for i := 0; i < len(layer)-1; i++ {
			diff := layer[i+1] - layer[i]
			newLayer = append(newLayer, diff)

			if diff == 0 {
				zeroes++
			}
		}

		layers = append(layers, newLayer)
		index++

		if len(newLayer) == zeroes {
			break
		}
	}

	for i := len(layers) - 2; i >= 0; i-- {
		toAdd := layers[i+1][len(layers[i+1])-1]
		value := layers[i][len(layers[i])-1]
		layers[i] = append(layers[i], value+toAdd)
	}

	return layers[0][len(layers[0])-1]
}

func solve(input string) (int, int) {
	data := tools.ReadLines(input)
	part1, part2 := 0, 0

	for _, line := range data {
		nums := tools.StrToInt(strings.Fields(line))
		part1 += extrapolate(nums)
		slices.SortStableFunc(nums, func(_, _ int) int { return -1 })
		part2 += extrapolate(nums)
	}

	return part1, part2
}

func main() {
	fmt.Println(solve("test.txt"))
	fmt.Println(solve(tools.GetData(2023, 9)))
}
