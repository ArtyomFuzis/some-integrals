import time
from typing import Callable

precision = 10e-9
bin_search_max_distance = 1


def generate_polyline(func: Callable[[float], float], start_point_x: float, final_point_x: float, distance: float) -> \
        list[tuple[float, float]]:
    global precision
    global bin_search_max_distance

    def next_polyline(cur_point: tuple[float, float]) -> tuple[float, float]:
        left = 0
        right = bin_search_max_distance
        while right - left > precision:
            mid = (left + right) / 2
            cur_distance = (mid ** 2 + (cur_point[1] - func(cur_point[0] + mid)) ** 2) ** 0.5
            if type(cur_distance) is complex:
                right = mid
            elif cur_distance < distance:
                left = mid
            else:
                right = mid
        return cur_point[0] + left, func(cur_point[0] + left)

    reverse = False
    if start_point_x > final_point_x:
        reverse = True
        start_point_x, final_point_x = final_point_x, start_point_x
    start_point = (start_point_x, func(start_point_x))
    vertexes = [start_point]
    while vertexes[-1][0] < final_point_x:
        vertexes.append(next_polyline(vertexes[-1]))
    vertexes[-1] = (final_point_x, func(final_point_x))
    if reverse:
        vertexes = vertexes[::-1]
    return vertexes


def make_partition_polyline(polyline: list[tuple[float, float]],
                            partition: Callable[[tuple[float, float], tuple[float, float]], tuple[float, float]]) -> \
        list[tuple[float, float]]:
    return [partition(polyline[i], polyline[i + 1]) for i in range(len(polyline) - 1)]


class IntegralParams:
    def __init__(self, partition: Callable[[tuple[float, float], tuple[float, float]], tuple[float, float]],
                 distance: float):
        self.partition = partition
        self.distance = distance


class Curve:
    def __init__(self, func: Callable[[float], float], x_start: float, x_final: float):
        self.func = func
        self.x_start = x_start
        self.x_final = x_final

    def calculate_curve_integral2(self, field: Callable[[tuple[float, float]], tuple[float, float]],
                                  params: IntegralParams) -> float:
        polyline = generate_polyline(self.func, self.x_start, self.x_final, params.distance)
        partition = make_partition_polyline(polyline, params.partition)
        res = 0
        for i in range(len(partition)):
            f_v = field(partition[i])
            res += (polyline[i + 1][0] - polyline[i][0]) * f_v[0] + (polyline[i + 1][1] - polyline[i][1]) * f_v[1]
        return res


def fixed_len_str(s: str, l: int) -> str:
    if len(s) > l:
        return "overflow"
    return s + " " * (l - len(s))


partition_left = lambda v1, v2: v1

field = lambda v: (v[0] ** 2 + v[1] ** 2, v[0] * v[1])
print(f'| Время: | Результат:     | Отклонение:    |')
for s in [0.1, 0.01, 0.001]:
    start_time = time.time()
    params = IntegralParams(partition_left, s)
    L1 = Curve(lambda x: (4 - x ** 2) ** 0.5, -2, 2)
    L2 = Curve(lambda x: 0, 2, -2)
    I1 = L1.calculate_curve_integral2(field, params)
    I2 = L2.calculate_curve_integral2(field, params)
    I = I1 + I2
    end = time.time()
    print(f'| {fixed_len_str(str(round(end - start_time, 3)), 5)}с | {fixed_len_str(str(round(I, 12)), 15)}| {fixed_len_str(str(round(16 / 3 - I, 12)), 15)}|')
