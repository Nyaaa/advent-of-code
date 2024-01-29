package main

import (
	"fmt"
	"aoc/tools"
	"strconv"
	"sort"
)

func solve(input string) (int, int) {
	lines := tools.ReadLines(input)
	blocks := tools.SplitBlocks(lines)
	var elves []int
	for _, elf := range(blocks) {
		value := 0
		for _, item := range(elf) {
			v, _ := strconv.Atoi(item)
			value += v
		}
		elves = append(elves, value)
	}
	sort.Ints(elves)
	top3 := elves[len(elves)-3:]
	part1 := top3[len(top3)-1]
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