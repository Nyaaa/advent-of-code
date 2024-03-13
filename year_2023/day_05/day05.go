package main

import (
	"aoc/tools"
	"fmt"
	"slices"
	"strings"
)

type mapping struct {
	destStart int
	srcStart  int
	length    int
}

func part1(ranges [][]mapping, seeds []int) int {
	result := make([]int, len(seeds))
	copy(result, seeds)

	for _, block := range ranges {
		for i, seed := range result {
			for _, m := range block {
				if m.srcStart <= seed && m.srcStart+m.length > seed {
					result[i] = m.destStart + seed - m.srcStart
				}
			}
		}
	}

	return slices.Min(result)
}

func part2(ranges [][]mapping, seeds []int) int {
	seed := 0

	for {
		seed++
		current := seed

		for i := len(ranges) - 1; i >= 0; i-- {
			maps := ranges[i]
			match := 0

			for _, m := range maps {
				if m.destStart <= current && m.destStart+m.length > current {
					match = m.srcStart + current - m.destStart
				}
			}

			if match != 0 {
				current = match
			}
		}

		for i := 0; i < len(seeds); i += 2 {
			start, length := seeds[i], seeds[i+1]
			if current > start && current < start+length {
				return seed
			}
		}
	}
}

func solve(input string) (int, int) {
	blocks := tools.SplitBlocks(tools.ReadLines(input))
	seeds := tools.StrToInt(strings.Fields(blocks[0][0])[1:])
	ranges := [][]mapping{}

	for _, block := range blocks[1:] {
		b := []mapping{}

		for _, line := range block[1:] {
			nums := tools.StrToInt(strings.Fields(line))
			b = append(b, mapping{nums[0], nums[1], nums[2]})
		}

		ranges = append(ranges, b)
	}

	return part1(ranges, seeds), part2(ranges, seeds)
}

func main() {
	fmt.Println(solve("test.txt"))
	fmt.Println(solve(tools.GetData(2023, 5)))
}
