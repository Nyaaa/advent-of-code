package main

import (
    "fmt"
    "aoc/tools"
	"strconv"
	"slices"
	"sort"
)

func main() {
	lines := tools.Read_lines("test.txt")
	blocks := tools.Split_blocks(lines)
	var elves []int
	for _, elf := range(blocks) {
		value := 0
		for _, item := range(elf) {
			v, _ := strconv.Atoi(item)
			value += v
		}
		elves = append(elves, value)
	}
	fmt.Println(slices.Max(elves))

	sort.Ints(elves)
	top3 := elves[len(elves)-3:]
	part2 := 0
	for i := 0; i < 3; i++ {
        part2 += top3[i]
    }

	fmt.Println(part2)
}