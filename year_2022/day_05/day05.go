package main

import (
	"aoc/tools"
	"fmt"
	"regexp"
	"unicode"
)

func solve(input []string) (int, int) {
	blocks := tools.SplitBlocks(input)
	part1 := 0
	part2 := 0
	load := map[int]string{}
	commands := [][]int{}

	for _, line := range blocks[0] {
		for index, char := range line {
			if unicode.IsLetter(char) {
				load[index] += string(char)
			}
		}
	}

	for _, line := range blocks[1] {
		digits := regexp.MustCompile(`\d+`)
		matches := digits.FindAllString(line, -1)
		commands = append(commands, tools.StrToInt(matches))
	}

	fmt.Println(load, commands)

	return part1, part2
}

func main() {
	testData := tools.ReadLines("test.txt")
	// data := tools.ReadLines(tools.GetData(2022, 05))

	fmt.Println(solve(testData))
	// fmt.Println(solve(data))
}
