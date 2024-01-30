package main

import (
	"fmt"
	"aoc/tools"
	"strings"
	mapset "github.com/deckarep/golang-set/v2"
)

func char_to_int(char string) int{
	val := int([]rune(char)[0])
	if strings.ToUpper(char) == char {
		val -= 38
	} else {val -= 96}
	return val
}

func part1(data []string) int {
	result := 0
	for _, line := range(data){
		if line == "" {continue}
		chars := strings.Split(line,"")
		mid := len(line) / 2
		left := mapset.NewSet[string](chars[:mid]...)
		right := mapset.NewSet[string](chars[mid:]...)
		common := left.Intersect(right)
		result += char_to_int(<-common.Iter())
	}
	return result
}

func part2(data []string) int {
	result := 0
	for i := 0; i < len(data) - 1; i += 3 {
		left := mapset.NewSet[string](strings.Split(data[i],"")...)
		mid := mapset.NewSet[string](strings.Split(data[i + 1],"")...)
		right := mapset.NewSet[string](strings.Split(data[i + 2],"")...)
		common := left.Intersect(mid)
		common = common.Intersect(right)
		result += char_to_int(<-common.Iter())
	}
	return result
}

func main() {
	data := tools.ReadLines(tools.GetData(2022, 03))
	test_data := []string{
	"vJrwpWtwJgWrhcsFMMfFFhFp",
	"jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL",
	"PmmdzqPrVvPwwTWBwg",
	"wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn",
	"ttgJtRGJQctTZtZT",
	"CrZsJsPPZsGzwwsLwLmpwMDw"}
	fmt.Println(part1(test_data), part2(test_data))
	fmt.Println(part1(data), part2(data))
}