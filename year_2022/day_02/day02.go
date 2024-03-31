package main

import (
	"github.com/Nyaaa/advent-of-code/tools"
	"fmt"
	"strings"
)

func checkWin(left int, right int) int {
	if left == right {
		return 3
	} else if (left == 3 && right == 1) || (left == 1 && right == 2) || left == 2 && right == 3 {
		return 6
	}

	return 0
}

func play(data []string) (int, int) {
	moves := map[string]int{"A": 1, "X": 1, "B": 2, "Y": 2, "C": 3, "Z": 3}
	outcomes := map[string]int{"X": 0, "Y": 3, "Z": 6}
	part1 := 0
	part2 := 0

	for _, line := range data {
		if line != "" {
			letters := strings.Fields(line)
			left := moves[letters[0]]
			right := moves[letters[1]]
			part1 += right + checkWin(left, right)

			for i := 1; i <= 3; i++ {
				p2 := checkWin(left, i)
				if p2 == outcomes[letters[1]] {
					part2 += i + p2
				}
			}
		}
	}

	return part1, part2
}

func main() {
	testData := []string{"A Y", "B X", "C Z"}
	data := tools.ReadLines(tools.GetData(2022, 02))

	fmt.Println(play(testData))
	fmt.Println(play(data))
}
