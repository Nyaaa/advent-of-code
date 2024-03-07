package main

import (
	"aoc/tools"
	"fmt"
	"regexp"
	"strconv"
)

func solve(input string) (int, int) {
	part1, part2 := 0, 0
	re := regexp.MustCompile(`(\d+) (\w+)`)
	nums := regexp.MustCompile(`\d+`)

	for _, line := range tools.ReadLines(input) {
		b := map[string]int{"red": 0, "green": 0, "blue": 0}
		cubes := re.FindAllStringSubmatch(line, -1)

		for _, cube := range cubes {
			value, _ := strconv.Atoi(cube[1])
			b[cube[2]] = max(value, b[cube[2]])
		}

		if b["red"] <= 12 && b["green"] <= 13 && b["blue"] <= 14 {
			index, _ := strconv.Atoi(nums.FindString(line))
			part1 += index
		}

		part2 += b["red"] * b["green"] * b["blue"]
	}

	return part1, part2
}

func main() {
	fmt.Println(solve("test.txt"))
	fmt.Println(solve(tools.GetData(2023, 2)))
}
