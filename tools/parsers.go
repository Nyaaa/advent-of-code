package tools

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
)

func ReadLines(input string) []string {
	file, err := os.Open(input)

	if err != nil {
		log.Fatal(err)
	}

	defer file.Close()

	lines := []string{}
	s := bufio.NewScanner(file)

	for s.Scan() {
    	lines = append(lines, s.Text())
	}

	return lines
}

func SplitBlocks(input []string) [][]string {
	output := [][]string{}
	temp := []string{}

	for _, item := range input {
		if item == "" {
			output = append(output, temp)
			temp = nil
		} else {
			temp = append(temp, item)
		}
	}

	if temp != nil {
		output = append(output, temp)
	}

	return output
}

func GetData(year int, day int) string {
	_, file, _, _ := runtime.Caller(0)
	basepath := filepath.Dir(file)

	return filepath.Join(basepath, "../", "aoc-inputs", strconv.Itoa(year), fmt.Sprintf("day%02d.txt", day))
}
