package main

import (
	"aoc/tools"
	"fmt"
	"sort"
	"strconv"
)

func solve(input string) (int, int) {
	lines := tools.ReadLines(input)
	blocks := tools.SplitBlocks(lines)
	elves := make([]int, len(blocks))

	for i, elf := range blocks {
		value := 0

		for _, item := range elf {
			v, _ := strconv.Atoi(item)
			value += v
		}

		elves[i] = value
	}

	sort.Ints(elves)
	top3 := elves[len(elves)-3:]
	part1 := top3[2]
	part2 := 0

	for i := 0; i < 3; i++ {
		part2 += top3[i]
	}

	return part1, part2
}

func main() {
	fmt.Println(solve("test.txt"))
	fmt.Println(solve(tools.GetData(2022, 01)))
}
