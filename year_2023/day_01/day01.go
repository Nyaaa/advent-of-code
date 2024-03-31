package main

import (
	"github.com/Nyaaa/advent-of-code/tools"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

func solve(input string, part2 bool) int {
	result := 0
	nums := regexp.MustCompile(`\d`)
	words := map[string]string{"one": "o1e", "two": "t2o", "three": "t3e",
		"four": "f4r", "five": "f5e", "six": "s6x",
		"seven": "s7n", "eight": "e8t", "nine": "n9e"}

	for _, line := range tools.ReadLines(input) {
		if part2 {
			for from, to := range words {
				line = strings.ReplaceAll(line, from, to)
			}
		}

		digits := nums.FindAllString(line, -1)
		value := fmt.Sprint(digits[0], digits[len(digits)-1])
		valueInt, _ := strconv.Atoi(value)
		result += valueInt
	}

	return result
}

func main() {
	data := tools.GetData(2023, 1)

	fmt.Println(solve("test.txt", false), solve("test2.txt", true))
	fmt.Println(solve(data, false), solve(data, true))
}
