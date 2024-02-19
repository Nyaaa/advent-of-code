package main

import (
	"aoc/tools"
	"fmt"
	"strconv"
	"strings"
)

func solve(input string, length int) int {
	rope := []complex128{}
	movements := map[string]complex128{"R": 1, "L": -1, "U": 1i, "D": -1i}
	visited := map[complex128]bool{}

	for i := 0; i < length; i++ {
		rope = append(rope, 0i)
	}

	for _, line := range tools.ReadLines(input) {
		directions := strings.Fields(line)
		distance, _ := strconv.Atoi(directions[1])

		for i := 0; i < distance; i++ {
			rope[0] += movements[directions[0]]

			for iSegment := 1; iSegment < len(rope); iSegment++ {
				headPosition := rope[iSegment-1]
				tailPosition := rope[iSegment]
				move := 0i

				switch headPosition - tailPosition {
				case 2:
					move = 1
				case -2:
					move = -1
				case 2i:
					move = 1i
				case -2i:
					move = -1i
				case 1 + 2i, 2 + 1i, 2 + 2i:
					move = 1 + 1i
				case 1 - 2i, 2 - 1i, 2 - 2i:
					move = 1 - 1i
				case -1 + 2i, -2 + 1i, -2 + 2i:
					move = -1 + 1i
				case -1 - 2i, -2 - 1i, -2 - 2i:
					move = -1 - 1i
				}

				newTailPos := tailPosition + move
				rope[iSegment] = newTailPos
				_, positionSeen := visited[newTailPos]

				if iSegment+1 == len(rope) && !positionSeen {
					visited[newTailPos] = true
				}
			}
		}
	}

	return len(visited)
}

func main() {
	fmt.Println(solve("test.txt", 2))
	fmt.Println(solve("test2.txt", 10))

	fmt.Println(solve(tools.GetData(2022, 9), 2))
	fmt.Println(solve(tools.GetData(2022, 9), 10))
}
