from tools import parsers, loader
import re
from typing import NamedTuple
from collections import deque


class Blueprint(NamedTuple):
    id: int
    ore: int  # ore
    clay: int  # ore
    obsidian: tuple[int, int]  # ore, clay
    geode: tuple[int, int]  # ore, obsidian

    def max_cost(self):
        """ore, clay, obsidian"""
        ore = max(self.ore, self.clay, self.obsidian[0], self.geode[0])
        return ore, self.obsidian[1], self.geode[1]


class State(NamedTuple):
    time: int
    resources: tuple = (0, 0, 0, 0)  # ore, clay, obsidian, geode
    robots: tuple = (1, 0, 0, 0)

    def value(self):
        return self.resources[3]


class Factory:
    def __init__(self, data):
        self.blueprints = []

        for line in data:
            nums = list(map(int, re.findall(r'\d+', line)))
            bp = Blueprint(id=nums[0], ore=nums[1], clay=nums[2], obsidian=(nums[3], nums[4]), geode=(nums[5], nums[6]))
            self.blueprints.append(bp)

    @staticmethod
    def prune(state: State, max_ore: int, max_clay: int, max_obsidian: int) -> State:
        ore = min(state.resources[0], state.time * max_ore - state.robots[0] * (state.time - 1))
        clay = min(state.resources[1], state.time * max_clay - state.robots[1] * (state.time - 1))
        obsidian = min(state.resources[2], state.time * max_obsidian - state.robots[2] * (state.time - 1))

        return State(state.time, (ore, clay, obsidian, state.resources[3]), state.robots)

    def evaluate(self, bp: Blueprint, state: State) -> int:
        value = 0
        queue = deque([state])
        done = set()
        max_ore, max_clay, max_obsidian = bp.max_cost()

        while queue:
            state = queue.popleft()
            value = max(value, state.value())
            state = self.prune(state, max_ore, max_clay, max_obsidian)

            if state in done or state.time == 0:
                continue
            done.add(state)
            time = state.time - 1
            production = tuple(map(sum, zip(state.resources, state.robots)))

            # pass a turn
            queue.append(State(time, production, state.robots))

            # ore robot
            if state.resources[0] >= bp.ore and state.robots[0] <= max_ore:
                res = (production[0] - bp.ore, production[1], production[2], production[3])
                rob = (state.robots[0] + 1, state.robots[1], state.robots[2], state.robots[3])
                queue.append(State(time, res, rob))

            # clay robot
            if state.resources[0] >= bp.clay and state.robots[1] <= max_clay:
                res = (production[0] - bp.clay, production[1], production[2], production[3])
                rob = (state.robots[0], state.robots[1] + 1, state.robots[2], state.robots[3])
                queue.append(State(time, res, rob))

            # obsidian robot
            if state.resources[0] >= bp.obsidian[0] and state.resources[1] >= bp.obsidian[1]\
                    and state.robots[2] <= max_obsidian:
                res = (production[0] - bp.obsidian[0], production[1] - bp.obsidian[1], production[2], production[3])
                rob = (state.robots[0], state.robots[1], state.robots[2] + 1, state.robots[3])
                queue.append(State(time, res, rob))

            # geode robot
            if state.resources[0] >= bp.geode[0] and state.resources[2] >= bp.geode[1]:
                res = (production[0] - bp.geode[0], production[1], production[2] - bp.geode[1], production[3])
                rob = (state.robots[0], state.robots[1], state.robots[2], state.robots[3] + 1)
                queue.append(State(time, res, rob))

        return value

    def part_1(self):
        return sum([bp.id * self.evaluate(bp, State(24)) for bp in self.blueprints])


print(Factory(parsers.lines('test.txt')).part_1())
# print(Factory(parsers.lines(loader.get())).part_1())  # 1262
