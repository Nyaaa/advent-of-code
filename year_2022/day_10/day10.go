package main

import (
	"aoc/tools"
	"fmt"
	"strconv"
	"strings"
)

func solve(input string) (int, string) {
	var (
		instructions                              = tools.ReadLines(input)
		register                                  = 1
		cycle, index, skip, addValue, part1, hPos int
		part2                                     string
	)

	for index < len(instructions) {
		cycle++
		hPos++

		if skip == 1 {
			register += addValue
		}

		if register <= hPos && hPos <= register+2 {
			part2 += "█"
		} else {
			part2 += "."
		}

		if cycle == 20 || (cycle+20)%40 == 0 {
			part1 += cycle * register
		} else if cycle%40 == 0 {
			part2 += "\n"
			hPos = 0
		}

		if skip == 2 {
			skip--

			continue
		}

		line := instructions[index]

		if line != "noop" {
			addValue, _ = strconv.Atoi(strings.Fields(line)[1])
			skip = 2
		} else {
			addValue = 0
		}

		index++
	}

	return part1, part2
}

func main() {
	data := [2]string{"test10.txt", tools.GetData(2022, 10)}

	for _, input := range data {
		part1, part2 := solve(input)
		fmt.Println(part1)
		fmt.Println(part2)
	}
}
