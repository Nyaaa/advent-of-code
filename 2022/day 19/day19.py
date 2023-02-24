from tools import parsers, loader
import re
from dataclasses import dataclass


@dataclass
class Blueprint:
    id: int
    ore: int  # ore
    clay: int  # ore
    obsidian: tuple[int, int]  # ore, clay
    geode: tuple[int, int]  # ore, obsidian
    value: int = 0


class Factory:
    def __init__(self, data):
        self.blueprints = []
        self.resources = {'ore': 0, 'clay': 0, 'obsidian': 0, 'geode': 0}
        self.robots = {'ore': 1, 'clay': 0, 'obsidian': 0, 'geode': 0}

        for line in data:
            nums = list(map(int, re.findall(r'\d+', line)))
            bp = Blueprint(id=nums[0], ore=nums[1], clay=nums[2],
                           obsidian=(nums[3], nums[4]), geode=(nums[5], nums[6]))
            self.blueprints.append(bp)

    def evaluate(self, bp: Blueprint):
        time = 1

        while time <= 24:
            self.produce()
            print('step', time)
            print('robots:', self.robots)
            print('resources:', self.resources)
            time += 1
            robot = self.build(bp)
            print(robot)
            if robot != 'pass':
                self.robots[robot] += 1


    def produce(self):
        for resource in self.resources:
            self.resources[resource] += self.robots[resource]

    def build(self, bp: Blueprint):
        return 'pass'

    def part_1(self):
        result = 0
        for bp in self.blueprints:
            result = self.evaluate(bp)
            print(result)
        return result


print(Factory(parsers.lines('test.txt')).part_1())
