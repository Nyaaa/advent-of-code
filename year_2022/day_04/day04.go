package main

import (
	"aoc/tools"
	"fmt"
	"regexp"
)

func isInRange(value int, min int, max int) bool {
	return value >= min && value <= max
}

func solve(input []string) (int, int) {
	part1 := 0
	noOverlaps := 0

	for _, line := range input {
		if line == "" {
			noOverlaps++

			continue
		}

		digits := regexp.MustCompile(`\d+`)
		elves := digits.FindAllString(line, -1)
		groups := tools.StrToInt(elves)

		if (isInRange(groups[0], groups[2], groups[3]) && isInRange(groups[1], groups[2], groups[3])) ||
			(isInRange(groups[2], groups[0], groups[1]) && isInRange(groups[3], groups[0], groups[1])) {
			part1++
		}

		if groups[1] < groups[2] || groups[3] < groups[0] {
			noOverlaps++
		}
	}

	return part1, len(input) - noOverlaps
}

func main() {
	testData := []string{"2-4,6-8", "2-3,4-5", "5-7,7-9", "2-8,3-7", "6-6,4-6", "2-6,4-8"}
	data := tools.ReadLines(tools.GetData(2022, 04))

	fmt.Println(solve(testData))
	fmt.Println(solve(data))
}
