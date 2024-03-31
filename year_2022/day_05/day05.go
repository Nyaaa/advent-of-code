package main

import (
	"github.com/Nyaaa/advent-of-code/tools"
	"fmt"
	"regexp"
	"strconv"
	"unicode"
)

func solve(input []string, part2 bool) string {
	blocks := tools.SplitBlocks(input)
	load := map[string]string{}
	indexMap := map[int]string{}

	for index, char := range blocks[0][len(blocks[0])-1] {
		indexMap[index] = string(char)
	}

	for _, line := range blocks[0] {
		for index, char := range line {
			if unicode.IsLetter(char) {
				load[indexMap[index]] += string(char)
			}
		}
	}

	for _, line := range blocks[1] {
		digits := regexp.MustCompile(`\d+`)
		matches := digits.FindAllString(line, -1)
		amount, _ := strconv.Atoi(matches[0])
		fromCol, toCol := load[matches[1]], load[matches[2]]
		block := ""

		if !part2 {
			for _, v := range fromCol[:amount] {
				block = string(v) + block
			}
		} else {
			block = fromCol[:amount]
		}

		load[matches[1]] = fromCol[amount:]
		load[matches[2]] = block + toCol
	}

	result := ""

	for i := 1; i <= len(load); i++ {
		result += string(load[strconv.Itoa(i)][0])
	}

	return result
}

func main() {
	testData := tools.ReadLines("test.txt")
	data := tools.ReadLines(tools.GetData(2022, 05))

	fmt.Println(solve(testData, false), solve(testData, true))
	fmt.Println(solve(data, false), solve(data, true))
}
