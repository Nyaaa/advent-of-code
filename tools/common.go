package tools

import (
	"strconv"
)

func StrToInt(strings []string) []int {
	output := []int{}

	for i := 0; i < len(strings); i++ {
		v, _ := strconv.Atoi(strings[i])
		output = append(output, v)
	}

	return output
}

func Counter(input string) map[rune]int {
	counts := make(map[rune]int)

	for _, char := range input {
		counts[char]++
	}

	return counts
}
