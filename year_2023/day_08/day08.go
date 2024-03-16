package main

import (
	"aoc/tools"
	"fmt"
	"regexp"
	"strings"
)

func move(location string, instructions string, mapping map[string][]string) int {
	steps := 0

	for !strings.HasSuffix(location, "Z") {
		if string(instructions[steps%len(instructions)]) == "L" {
			location = mapping[location][0]
		} else {
			location = mapping[location][1]
		}

		steps++
	}

	return steps
}

func solve(input string, part2 bool) int {
	data := tools.SplitBlocks(tools.ReadLines(input))
	nodes := regexp.MustCompile(`\w+`)
	mapping := map[string][]string{}
	starts := []string{}
	stepCounts := []int{}

	if !part2 {
		starts = append(starts, "AAA")
	}

	for _, line := range data[1] {
		values := strings.Split(line, " = ")
		mapping[values[0]] = nodes.FindAllString(values[1], -1)

		if part2 && strings.HasSuffix(values[0], "A") {
			starts = append(starts, values[0])
		}
	}

	for _, start := range starts {
		stepCounts = append(stepCounts, move(start, data[0][0], mapping))
	}

	return tools.LCM(stepCounts...)
}

func main() {
	data := tools.GetData(2023, 8)

	fmt.Println(solve("test.txt", false), solve("test2.txt", true))
	fmt.Println(solve(data, false), solve(data, true))
}
