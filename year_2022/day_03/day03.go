package main

import (
	"github.com/Nyaaa/advent-of-code/tools"
	"fmt"
)

func charToInt(val byte) int {
	if val < 95 {
		val -= 38
	} else {
		val -= 96
	}

	return int(val)
}

func intersect(left, right []byte) []byte {
	seen := map[byte]bool{}
	result := []byte{}

	for _, char := range left {
		seen[char] = true
	}

	for _, char := range right {
		if seen[char] {
			result = append(result, char)
		}
	}

	return result
}

func part1(data []string) int {
	result := 0

	for _, line := range data {
		mid := len(line) / 2
		left := []byte(line[:mid])
		right := []byte(line[mid:])
		common := intersect(left, right)
		result += charToInt(common[0])
	}

	return result
}

func part2(data []string) int {
	result := 0

	for i := 0; i < len(data)-1; i += 3 {
		left := []byte(data[i])
		mid := []byte(data[i+1])
		right := []byte(data[i+2])
		common := intersect(left, mid)
		common = intersect(common, right)
		result += charToInt(common[0])
	}

	return result
}

func main() {
	data := tools.ReadLines(tools.GetData(2022, 03))
	testData := tools.ReadLines("test.txt")
	fmt.Println(part1(testData), part2(testData))
	fmt.Println(part1(data), part2(data))
}
