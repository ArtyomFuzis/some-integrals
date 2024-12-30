import time
from math import floor
from typing import Callable


def generate_grid(distance: float, min_border: tuple[float, float], max_border: tuple[float, float]) -> (
        tuple)[list[float], list[float]]:
    return ([min_border[0] + distance * i for i in
             range(floor((max_border[0] - min_border[0]) / distance) + 1)]
            , [min_border[1] + distance * i for i in
               range(floor((max_border[1] - min_border[1]) / distance) + 1)])


def make_partition_grid(grid: tuple[list[float], list[float]], distance: float,
                        partition: Callable[[tuple[float, float], float], tuple[float, float]]) -> (
        list)[tuple[float, float]]:
    res = []
    for i in grid[0]:
        for j in grid[1]:
            res.append(partition((i, j), distance))
    return res


class IntegralParams:
    def __init__(self, partition: Callable[[tuple[float, float], float], tuple[float, float]],
                 distance: float):
        self.partition = partition
        self.distance = distance


class Area:
    def __init__(self, predicate: Callable[[tuple[float, float]], bool], min_border: tuple[float, float],
                 max_border: tuple[float, float]):
        self.predicate = predicate
        self.min_border = min_border
        self.max_border = max_border

    def calculate_area_integral(self, func: Callable[[tuple[float, float]], float], params: IntegralParams) -> float:
        grid = generate_grid(params.distance, self.min_border, self.max_border)
        partition = make_partition_grid(grid, params.distance, params.partition)
        res = 0
        for el in partition:
            if self.predicate(el):
                res += func(el) * params.distance ** 2
        return res


def fixed_len_str(s: str, l: int) -> str:
    if len(s) > l:
        return "overflow"
    return s + " " * (l - len(s))


partition_center = lambda vl, distance: (vl[0] + distance / 2, vl[1] + distance / 2)
grin_func = lambda v: v[1]

print(f'| Время: | Результат:     | Отклонение:    |')
for s in [0.1, 0.01, 0.001]:
    start_time = time.time()
    params = IntegralParams(partition_center, s)
    D = Area(lambda v: v[0] ** 2 + v[1] ** 2 <= 4 and v[1] >= 0, (-2, 0), (2, 2))
    I = D.calculate_area_integral(grin_func, params)
    end = time.time()
    print(f'| {fixed_len_str(str(round(end - start_time, 3)), 5)}с | {fixed_len_str(str(round(I, 12)), 15)}| {fixed_len_str(str(round(16 / 3 - I, 12)), 15)}|')
