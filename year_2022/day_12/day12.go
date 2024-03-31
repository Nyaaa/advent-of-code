package main

import (
	"github.com/Nyaaa/advent-of-code/tools"
	"fmt"
	"math"
)

type state struct {
	location complex128
	steps    int
}

func solve(input string, start rune) int {
	data := tools.ReadLines(input)
	nodes := map[complex128]rune{}
	heights := map[rune]int{}
	heights['S'] = 0
	heights['E'] = 25
	adjacent := [4]complex128{-1, 1, -1i, 1i}
	queue := []state{}
	seen := map[complex128]bool{}
	shortest := math.MaxInt

	for i := 'a'; i <= 'z'; i++ {
		heights[i] = int(i) - 97
	}

	for i, row := range data {
		for j, letter := range row {
			n := complex(float64(i), float64(j))
			nodes[n] = letter

			if letter == start {
				queue = append(queue, state{n, 0})
			}
		}
	}

	for len(queue) > 0 {
		current := queue[0]
		curRune := nodes[current.location]
		curLevel := heights[curRune]
		queue = queue[1:]

		if curRune == 'E' {
			shortest = min(shortest, current.steps)
		}

		if seen[current.location] {
			continue
		}

		seen[current.location] = true

		for _, adj := range adjacent {
			adjLoc := current.location + adj
			adjRune, ok := nodes[adjLoc]
			adjLevel := heights[adjRune]

			if ok && adjLevel-curLevel <= 1 {
				queue = append(queue, state{adjLoc, current.steps + 1})
			}
		}
	}

	return shortest
}

func main() {
	testData := "test.txt"
	data := tools.GetData(2022, 12)

	fmt.Println(solve(testData, 'S'), solve(testData, 'a'))
	fmt.Println(solve(data, 'S'), solve(data, 'a'))
}
