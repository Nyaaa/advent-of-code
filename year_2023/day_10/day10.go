package main

import (
	"fmt"
	"github.com/Nyaaa/advent-of-code/tools"
	"strings"
)

type cart struct {
	location, start, direction complex128
}

func (c *cart) rotate(tile string) {
	switch tile {
	case "L", "7":
		if real(c.direction) == 0 {
			c.direction *= -1i
		} else {
			c.direction *= 1i
		}
	case "J", "F":
		if real(c.direction) == 0 {
			c.direction *= 1i
		} else {
			c.direction *= -1i
		}
	}
}

func getTrack(c cart, pipes map[complex128]string) []complex128 {
	track := []complex128{}

	for {
		track = append(track, c.location)
		c.location += c.direction
		c.rotate(pipes[c.location])

		if c.location == c.start {
			break
		}
	}

	return track
}

func part2(track []complex128) int {
	area := 0.0

	for i := 0; i < len(track); i++ {
		area += (imag(track[i]) * real(track[(i+1)%len(track)]))
		area -= (real(track[i]) * imag(track[(i+1)%len(track)]))
	}

	if area < 0 {
		area = -area
	}

	return int(area/2) - (len(track) / 2) + 1
}

func solve(input string) (int, int) {
	data := tools.ReadLines(input)
	pipes := map[complex128]string{}
	pipeChars := [7]string{"-", "|", "F", "7", "J", "L", "S"}
	cart := cart{0, 0, 0}

	for i, line := range data {
		for j, tile := range strings.Split(line, "") {
			location := complex(float64(i), float64(j))

			for _, char := range pipeChars {
				if char == tile {
					pipes[location] = tile

					if tile == "S" {
						cart.location = location
						cart.start = location
					}
				}
			}

			pipes[location] = tile
		}
	}

	for _, i := range [4]complex128{1, -1, 1i, -1i} {
		// This is potentially fragile if there are adjacent pipes that don't connect to S.
		_, ok := pipes[cart.location+i]

		if ok {
			cart.direction = i

			break
		}
	}

	track := getTrack(cart, pipes)

	return len(track) / 2, part2(track)
}

func main() {
	fmt.Println(solve("test3.txt"))
	fmt.Println(solve(tools.GetData(2023, 10)))
}
