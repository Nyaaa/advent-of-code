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
