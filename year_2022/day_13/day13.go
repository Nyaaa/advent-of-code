package main

import (
	"github.com/Nyaaa/advent-of-code/tools"
	"encoding/json"
	"fmt"
	"sort"
)

func unmarshal(str string) []any {
	result := []any{}
	_ = json.Unmarshal([]byte(str), &result)

	return result
}

func compare(left, right any) int {
	leftList, _ := left.([]any)
	rightList, _ := right.([]any)
	leftNum, leftIsNum := left.(float64)
	rightNum, rightIsNum := right.(float64)

	switch {
	case leftIsNum && rightIsNum:
		return int(leftNum - rightNum)
	case leftIsNum:
		leftList = []any{left}
	case rightIsNum:
		rightList = []any{right}
	}

	for i := 0; i < len(leftList) && i < len(rightList); i++ {
		if res := compare(leftList[i], rightList[i]); res != 0 {
			return res
		}
	}

	return len(leftList) - len(rightList)
}

func solve(input string) (int, int) {
	data := tools.SplitBlocks(tools.ReadLines(input))
	part1, part2 := 0, 1
	dataFull := []any{[]any{[]any{2.0}}, []any{[]any{6.0}}}

	for i, pair := range data {
		left := unmarshal(pair[0])
		right := unmarshal(pair[1])
		dataFull = append(dataFull, left, right)

		if compare(left, right) <= 0 {
			part1 += i + 1
		}
	}

	sort.Slice(dataFull, func(i, j int) bool {
		return compare(dataFull[i], dataFull[j]) <= 0
	})

	for i, line := range dataFull {
		if fmt.Sprint(line) == "[[2]]" || fmt.Sprint(line) == "[[6]]" {
			part2 *= i + 1
		}
	}

	return part1, part2
}

func main() {
	testData := "test13.txt"
	data := tools.GetData(2022, 13)

	fmt.Println(solve(testData))
	fmt.Println(solve(data))
}
