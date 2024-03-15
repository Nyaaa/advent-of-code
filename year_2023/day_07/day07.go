package main

import (
	"aoc/tools"
	"cmp"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

type hand struct {
	cards    []int
	bid      int
	handType int
}

func (h *hand) getHandType() {
	counts := map[int]int{}

	for _, card := range h.cards {
		counts[card]++
	}

	jokers := counts[1]
	if jokers < 5 {
		delete(counts, 1)
	}

	keys := make([]int, 0, len(counts))

	for key := range counts {
		keys = append(keys, key)
	}

	slices.SortFunc(keys, func(i, j int) int {
		return cmp.Compare(counts[j], counts[i])
	})

	if jokers > 0 && jokers < 5 {
		counts[keys[0]] += jokers
	}

	switch counts[keys[0]] {
	case 5:
		h.handType = 6 // five of a kind
	case 4:
		h.handType = 5 // four of a kind
	case 3:
		if counts[keys[1]] == 2 {
			h.handType = 4 // full house
		} else {
			h.handType = 3 // three of a kind
		}
	case 2:
		if counts[keys[1]] == 2 {
			h.handType = 2 // two pairs
		} else {
			h.handType = 1 // one pair
		}
	}
}

func solve(input string, part2 bool) int {
	data := tools.ReadLines(input)
	result := 0
	cardValues := map[string]int{}
	hands := []hand{}

	for i, char := range "23456789TJQKA" {
		cardValues[string(char)] = i + 2
	}

	if part2 {
		cardValues["J"] = 1
	}

	for _, line := range data {
		values := strings.Fields(line)
		bid, _ := strconv.Atoi(values[1])
		cards := []int{}

		for _, c := range values[0] {
			cards = append(cards, cardValues[string(c)])
		}

		h := hand{cards, bid, 0}
		h.getHandType()
		hands = append(hands, h)
	}

	slices.SortFunc(hands, func(i, j hand) int {
		if i.handType != j.handType {
			return cmp.Compare(i.handType, j.handType)
		}

		for l, leftCard := range i.cards {
			rightCard := j.cards[l]
			if leftCard != rightCard {
				return cmp.Compare(leftCard, rightCard)
			}
		}

		return 0
	})

	for i, hand := range hands {
		result += (i + 1) * hand.bid
	}

	return result
}

func main() {
	data := tools.GetData(2023, 7)

	fmt.Println(solve("test.txt", false), solve("test.txt", true))
	fmt.Println(solve(data, false), solve(data, true))
}
