# Advent of Code 2022
https://adventofcode.com/2022

### Day 1: Calorie Counting
* Just counting. Lists make it trivial!

### Day 2: Rock Paper Scissors
* Derp. Wasted a lot of time debugging. It was a typo.

### Day 3: Rucksack Reorganization
* Intersections.

### Day 4: Camp Cleanup
* More intersections.

### Day 5: Supply Stacks
* Simple, but fun! Used a LIFO queue.
* Refactor?

### Day 6: Tuning Trouble
* Easy, just basic compares.

### Day 7: No Space Left On Device
* This can probably be done in a much simpler way.

### Day 8: Treetop Tree House
* Made a grid, compared every node vertically and horisontally.
* Make a heat map?

### Day 9: Rope Bridge
* Part 1 was fairly simple, but figuring out and debugging correct diagonal motion for part 2 took a while.

### Day 10: Cathode-Ray Tube
* This is one of my favourites. So cool!

### Day 11: Monkey in the Middle
* Part 1 was trivial, spent a long time writing a parser. Should've considered hardcoding. 
* Had to look up solutions for part 2. WTF is modulo?

### Day 12: Hill Climbing Algorithm
* This one took forever.
* Attempt #1: Put nodes in a class, calculating connections is too slow.
* Attempt #2: Parsed the input into a dictionary representing a weighted graph to use a Dijkstra snippet I had. It worked for test data, failed on actual input. Must be a bug in my algorithm implementation.
* Attempt #3: Gave up, used networkx instead.
* Revisit this one?

### Day 13: Distress Signal
* Eval? Why not, indeed?
* Flattened the input lists for part 2. Can this approach be used for part 1?

### Day 14: Regolith Reservoir
* Pretty straightforward, used a numpy array, switched to chararray for better visuals.
* Make a gif? Colour?

### Day 15: Beacon Exclusion Zone
* Attempt #1: Apparently, brute force is not much of an option.
* Well, with this many cells, iterating through even a single row in kinda brutforcey.
* It's slow.
* Attempt #2: Welp, it ain't working. Original plan was to look for intersections of coverage areas and check along the borders, but I couldn't make a sensible implementation. 
* Attempt #3: Starting from scratch. Cheating with shapely.