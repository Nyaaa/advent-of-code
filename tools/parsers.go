package tools

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"unicode"
)

func ReadLines(input string) []string {
	b, err := os.ReadFile(input)

	if err != nil {
		log.Fatal(err)
	}

	return strings.Split(string(b), "\n")
}

func ReadString(input string) string {
	b, err := os.ReadFile(input)

	if err != nil {
		log.Fatal(err)
	}

	clean := strings.TrimFunc(string(b), func(r rune) bool {
		return !unicode.IsGraphic(r)
	})

	return clean
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

	return output
}

func GetData(year int, day int) string {
	_, file, _, _ := runtime.Caller(0)
	basepath := filepath.Dir(file)

	return filepath.Join(basepath, "../", "aoc-inputs", strconv.Itoa(year), fmt.Sprintf("day%02d.txt", day))
}
