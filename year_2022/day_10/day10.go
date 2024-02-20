package main

import (
	"aoc/tools"
	"fmt"
	"strconv"
	"strings"
)

func solve(input string) int {
	instructions := tools.ReadLines(input)
	x := 1
	cycle := 0
	index := 0
	skip := 0
	value := 0
	part1 := 0

	for {
		cycle++

		if skip == 1 {
			x += value
		}

		if cycle == 20 || (cycle+20)%40 == 0 {
			part1 += cycle * x
		}

		if skip == 2 {
			skip = 1

			continue
		}

		if index >= len(instructions) {
			break
		}

		line := instructions[index]

		if line != "noop" {
			value, _ = strconv.Atoi(strings.Fields(line)[1])
			skip = 2
		} else {
			value = 0
		}

		index++
	}

	return part1
}

func main() {
	fmt.Println(solve("test10.txt"))

	fmt.Println(solve(tools.GetData(2022, 10)))
}
