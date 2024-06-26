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

func GCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}

	return a
}

func LCM(integers ...int) int {
	a := integers[0]
	if len(integers) == 1 {
		return a
	}

	b := integers[1]
	result := a * b / GCD(a, b)

	for _, integer := range integers[2:] {
		result = LCM(result, integer)
	}

	return result
}

func Abs(number int) int {
	if number < 0 {
		return -number
	}

	return number
}

func MinMax(integers ...int) (int, int) {
	min, max := integers[0], integers[0]

	for _, i := range integers {
		if i > max {
			max = i
		} else if i < min {
			min = i
		}
	}

	return min, max
}
