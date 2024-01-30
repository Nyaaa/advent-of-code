package main

import (
	"fmt"
	"aoc/tools"
	"strings"
)

var (
	test_data = []string{"A Y", "B X", "C Z"}
	moves = map[string]int{"A": 1, "X": 1, "B": 2, "Y": 2, "C": 3, "Z": 3}
	outcomes = map[string]int{"X": 0, "Y": 3, "Z": 6}
)

func check_win(left int, right int) int {
	if left == right {
		return 3
	} else if (left == 3 && right == 1) || (left == 1 && right == 2) || left == 2 && right == 3 {
		return 6
	}
	return 0
}

func play(data []string) (int, int) {
	part1 := 0
	part2 := 0
	for _, line := range(data) {
		if line != "" {
			ln := strings.Fields(line)
			left := moves[ln[0]]
			right := moves[ln[1]]
			part1 += right + check_win(left, right)
			for i := 1; i <= 3; i++ {
				p2 := check_win(left, i)
				if p2 == outcomes[ln[1]] {
					part2 += i + p2
				}
			}
		}
	}
	return part1, part2
}

func main() {
	data := tools.ReadLines(tools.GetData(2022, 02))
	fmt.Println(play(test_data))
	fmt.Println(play(data))
}