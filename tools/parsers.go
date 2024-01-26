package tools

import (
	"log"
	"os"
	"strings"
)

func Read_lines(input string) []string {
    b, err := os.ReadFile(input)
    if err != nil {
        log.Fatal(err)
    }
    return strings.Split(string(b), "\n")
}

func Split_blocks(input []string) [][]string {
    var output [][]string
    var temp []string
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