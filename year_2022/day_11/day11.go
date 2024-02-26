package main

import (
	"aoc/tools"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

type monkey struct {
	inventory      []int
	operation      string
	operationValue string
	testValue      int
	targetTrue     int
	targetFalse    int
}

func (m monkey) action() {
	fmt.Println(m)
}

func solve(input [][]string, rounds int) int {
	monkeys := []monkey{}
	numbers := regexp.MustCompile(`\d+`)

	for _, block := range input {
		testValue, _ := strconv.Atoi(strings.Fields(block[3])[3])
		targetTrue, _ := strconv.Atoi(strings.Fields(block[4])[5])
		targetFalse, _ := strconv.Atoi(strings.Fields(block[5])[5])
		m := monkey{
			inventory:      tools.StrToInt(numbers.FindAllString(block[1], -1)),
			operation:      strings.Fields(block[2])[4],
			operationValue: strings.Fields(block[2])[5],
			testValue:      testValue,
			targetTrue:     targetTrue,
			targetFalse:    targetFalse,
		}
		monkeys = append(monkeys, m)
	}

	for i := 0; i < rounds; i++ {
		for _, m := range monkeys {
			m.action()
		}
	}

	return 0
}

func main() {
	testData := tools.SplitBlocks(tools.ReadLines("test11.txt"))

	fmt.Println(solve(testData, 20))
}
