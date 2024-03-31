package main

import (
	"github.com/Nyaaa/advent-of-code/tools"
	"fmt"
	"strconv"
	"strings"
)

func solve(input string, part2 bool) int {
	data := tools.ReadLines(input)
	times := strings.Fields(data[0])[1:]
	distances := strings.Fields(data[1])[1:]
	wins := 1

	if part2 {
		times = []string{strings.Join(times, "")}
		distances = []string{strings.Join(distances, "")}
	}

	for i := 0; i < len(times); i++ {
		time, _ := strconv.Atoi(times[i])
		distance, _ := strconv.Atoi(distances[i])
		w := 0

		for i := 1; i <= time+1; i++ {
			if distance < (time-i)*i {
				w++
			}
		}

		wins *= w
	}

	return wins
}

func main() {
	data := tools.GetData(2023, 6)

	fmt.Println(solve("test.txt", false), solve("test.txt", true))
	fmt.Println(solve(data, false), solve(data, true))
}
