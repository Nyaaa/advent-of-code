package main

import (
	"aoc/tools"
	"fmt"
)

func solve(input []string, length int) int {
	data := input[0]
	for i := 0; i <= len(data)-length; i++ {
		counts := tools.Counter(data[i : i+length])
		isValid := true

		for _, count := range counts {
			if count > 1 {
				isValid = false
			}
		}

		if isValid {
			return i + length
		}
	}

	return 0
}

func main() {
	testData := []string{"zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw"}
	data := tools.ReadLines(tools.GetData(2022, 06))

	fmt.Println(solve(testData, 4), solve(testData, 14))
	fmt.Println(solve(data, 4), solve(data, 14))
}
