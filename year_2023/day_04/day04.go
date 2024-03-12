package main

import (
	"aoc/tools"
	"fmt"
	"math"
	"strings"
)

type card struct {
	winning []int
	ticket  []int
}

func (c card) intersect() int {
	result := 0
	winning := map[int]bool{}

	for _, num := range c.winning {
		winning[num] = true
	}

	for _, val := range c.ticket {
		if winning[val] {
			result++
		}
	}

	return result
}

func solvePart1(tickets map[int]card) int {
	var result int

	for _, ticket := range tickets {
		if matches := ticket.intersect(); matches > 0 {
			result += int(math.Pow(2, float64(matches-1)))
		}
	}

	return result
}

func solvePart2(tickets map[int]card) int {
	result := 0
	matches := map[int][]int{}
	queue := []int{}

	for index := range tickets {
		queue = append(queue, index)
	}

	for len(queue) > 0 {
		current := queue[0]
		result++
		queue = queue[1:]
		wins, ok := matches[current]

		if !ok {
			for i := current; i < current+tickets[current-1].intersect(); i++ {
				wins = append(wins, i+1)
			}

			matches[current] = wins
		}

		queue = append(queue, wins...)
	}

	return result
}

func solve(input string) (int, int) {
	tickets := map[int]card{}

	for i, line := range tools.ReadLines(input) {
		split := strings.FieldsFunc(line, func(r rune) bool { return r == ':' || r == '|' })
		tickets[i+1] = card{
			winning: tools.StrToInt(strings.Fields(split[1])),
			ticket:  tools.StrToInt(strings.Fields(split[2])),
		}
	}

	return solvePart1(tickets), solvePart2(tickets)
}

func main() {
	fmt.Println(solve("test.txt"))
	fmt.Println(solve(tools.GetData(2023, 4)))
}
