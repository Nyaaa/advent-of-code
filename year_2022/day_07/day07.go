package main

import (
	"github.com/Nyaaa/advent-of-code/tools"
	"fmt"
	"math"
	"path"
	"strconv"
	"strings"
)

func getDirSizes(data []string) (map[string][]string, map[string]int) {
	var (
		currPath  string
		dirTree   = map[string][]string{}
		fileSizes = map[string]int{}
	)

	for _, line := range data {
		if line == "" {
			continue
		}

		splitLine := strings.Fields(line)
		fileSize, sizeErr := strconv.Atoi(splitLine[0])

		switch {
		case strings.HasPrefix(line, "$ cd"):
			folder := splitLine[2]
			if folder == ".." {
				currPath = path.Dir(currPath)
			} else {
				currPath = path.Join(currPath, folder)
				fileSizes[currPath] = 0
			}
		case sizeErr == nil:
			path := path.Join(currPath, splitLine[1])
			dirTree[currPath] = append(dirTree[currPath], path)
			fileSizes[path] = fileSize
		case strings.HasPrefix(line, "dir"):
			folder := splitLine[1]
			dirTree[currPath] = append(dirTree[currPath], path.Join(currPath, folder))
		}
	}

	return dirTree, calcSizes("/", dirTree, fileSizes)
}

func calcSizes(current string, dirTree map[string][]string, fileSizes map[string]int) map[string]int {
	size := 0

	for _, sub := range dirTree[current] {
		if fileSizes[sub] == 0 {
			fileSizes = calcSizes(sub, dirTree, fileSizes)
		}

		size += fileSizes[sub]
	}

	fileSizes[current] = size

	return fileSizes
}

func solve(file string) (int, int) {
	dirTree, fileSizes := getDirSizes(tools.ReadLines(file))
	part1 := 0
	part2 := math.MaxInt
	total := 70000000
	needed := 30000000
	free := total - fileSizes["/"]
	target := needed - free

	for folder := range dirTree {
		size := fileSizes[folder]
		if size <= 100000 {
			part1 += size
		} else if size >= target {
			part2 = min(part2, size)
		}
	}

	return part1, part2
}

func main() {
	fmt.Println(solve("test07.txt"))
	fmt.Println(solve(tools.GetData(2022, 07)))
}
