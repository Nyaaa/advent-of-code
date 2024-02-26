package main

import (
	"aoc/tools"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type monkey struct {
	inventory      []int
	operation      string
	operationValue string
	testValue      int
	targets        map[bool]int
	inspections    int
}

func (m monkey) action(monkeys []monkey, relief bool, modulo int) []monkey {
	for _, item := range m.inventory {
		value, err := strconv.Atoi(m.operationValue)

		if err != nil {
			value = item
		}

		if m.operation == "*" {
			item *= value
		} else {
			item += value
		}

		if relief {
			item /= 3
		} else {
			item %= modulo
		}

		target := m.targets[item%m.testValue == 0]
		monkeys[target].inventory = append(monkeys[target].inventory, item)
	}

	return monkeys
}

func solve(input string, rounds int, relief bool) int {
	data := tools.SplitBlocks(tools.ReadLines(input))
	monkeys := []monkey{}
	numbers := regexp.MustCompile(`\d+`)
	modulo := 1

	for _, block := range data {
		testValue, _ := strconv.Atoi(strings.Fields(block[3])[3])
		targetTrue, _ := strconv.Atoi(strings.Fields(block[4])[5])
		targetFalse, _ := strconv.Atoi(strings.Fields(block[5])[5])
		m := monkey{
			inventory:      tools.StrToInt(numbers.FindAllString(block[1], -1)),
			operation:      strings.Fields(block[2])[4],
			operationValue: strings.Fields(block[2])[5],
			testValue:      testValue,
			targets:        map[bool]int{true: targetTrue, false: targetFalse},
			inspections:    0,
		}
		monkeys = append(monkeys, m)
		modulo *= m.testValue
	}

	for i := 0; i < rounds; i++ {
		for j, m := range monkeys {
			monkeys = m.action(monkeys, relief, modulo)
			monkeys[j].inspections += len(m.inventory)
			monkeys[j].inventory = nil
		}
	}

	sort.Slice(monkeys, func(i, j int) bool {
		return monkeys[i].inspections > monkeys[j].inspections
	})

	return monkeys[0].inspections * monkeys[1].inspections
}

func main() {
	testData := "test11.txt"
	data := tools.GetData(2022, 11)

	fmt.Println(solve(testData, 20, true), solve(testData, 10000, false))
	fmt.Println(solve(data, 20, true), solve(data, 10000, false))
}
