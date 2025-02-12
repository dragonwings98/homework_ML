import numpy as np
import random
import math
import re
import time


def load_vrp_data(file_path):
    with open(file_path, "r") as file:
        vrp_data = [line.strip() for line in file.readlines()]

    node_coord_start = vrp_data.index("NODE_COORD_SECTION") + 1
    demand_start = vrp_data.index("DEMAND_SECTION") + 1
    depot_start = vrp_data.index("DEPOT_SECTION") + 1
    capacity_line = [line for line in vrp_data if "CAPACITY" in line][0]
    
    node_coords = {}
    for line in vrp_data[node_coord_start:demand_start - 1]:
        parts = list(map(int, re.findall(r'\d+', line)))
        node_coords[parts[0]] = (parts[1], parts[2])

    demands = {}
    for line in vrp_data[demand_start:depot_start - 1]:
        parts = list(map(int, re.findall(r'\d+', line)))
        demands[parts[0]] = parts[1]

    depot_id = int(re.findall(r'\d+', vrp_data[depot_start])[0])
    capacity = int(re.findall(r'\d+', capacity_line)[0])

    num_nodes = len(node_coords)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(1, num_nodes + 1):
        for j in range(1, num_nodes + 1):
            x1, y1 = node_coords[i]
            x2, y2 = node_coords[j]
            distance_matrix[i-1, j-1] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return depot_id - 1, list(range(1, num_nodes)), capacity, demands, distance_matrix


def load_optimal_solution(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    cost_line = [line for line in lines if line.startswith("Cost")][0]
    optimal_cost = int(re.findall(r'\d+', cost_line)[0])
    return optimal_cost


# 初始化参数
temp_initial = 1000  # 初始温度
temp_final = 1  # 终止温度
cooling_rate = 0.995  # 冷却率
max_iterations = 500  # 每个温度的迭代次数


def total_distance(route, distance_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i], route[i + 1]]
    return distance


def savings(depot, i, j, distance_matrix):
    return distance_matrix[depot][i] + distance_matrix[depot][j] - distance_matrix[i][j]


def initial_solution(depot, customers, capacity, demands, distance_matrix):
    # 初始化每个客户为一条单独的路线
    routes = [[depot, customer, depot] for customer in customers]
    # 计算所有节点对之间的节约值
    savings_list = []
    for i in customers:
        for j in customers:
            if i != j:
                s = savings(depot, i, j, distance_matrix)
                savings_list.append((s, i, j))
    # 按照节约值从大到小排序
    savings_list.sort(reverse=True)

    # 依次尝试合并节点对
    for s, i, j in savings_list:
        route_i = None
        route_j = None
        for route in routes:
            if i in route and route.index(i) != 0 and route.index(i) != len(route) - 1:
                route_i = route
            if j in route and route.index(j) != 0 and route.index(j) != len(route) - 1:
                route_j = route
        if route_i and route_j and route_i != route_j:
            # 检查合并后的路线是否超载
            # print(route_i)
            # print(route_j)
            load_i = sum(demands[node + 1] for node in route_i if node != depot)
            load_j = sum(demands[node + 1] for node in route_j if node != depot)
            if load_i + load_j <= capacity:
                # 合并路线
                if route_i[-2] == i and route_j[1] == j:
                    new_route = route_i[:-1] + route_j[1:]
                    routes.remove(route_i)
                    routes.remove(route_j)
                    routes.append(new_route)
    return routes


def swap(solution):
    new_solution = [route[:] for route in solution]
    route_idx = random.randint(0, len(new_solution) - 1)
    if len(new_solution[route_idx]) > 3:
        i, j = random.sample(range(1, len(new_solution[route_idx]) - 1), 2)
        new_solution[route_idx][i], new_solution[route_idx][j] = new_solution[route_idx][j], new_solution[route_idx][i]
    return new_solution


def simulated_annealing(depot, customers, capacity, demands, distance_matrix):
    current_solution = initial_solution(depot, customers, capacity, demands, distance_matrix)
    current_cost = sum(total_distance(route, distance_matrix) for route in current_solution)
    best_solution, best_cost = current_solution, current_cost
    temp = temp_initial

    while temp > temp_final:
        for _ in range(max_iterations):
            new_solution = swap(current_solution)
            new_cost = sum(total_distance(route, distance_matrix) for route in new_solution)

            if new_cost < current_cost or math.exp((current_cost - new_cost) / temp) > random.random():
                current_solution, current_cost = new_solution, new_cost
                if new_cost < best_cost:
                    best_solution, best_cost = new_solution, new_cost
        temp *= cooling_rate

    return best_solution, best_cost


def run_app(file_path, optimal_solution_path):

    start_time = time.time()


    depot, customers, capacity, demands, distance_matrix = load_vrp_data(file_path)
    best_routes, best_distance = simulated_annealing(depot, customers, capacity, demands, distance_matrix)
    optimal_cost = load_optimal_solution(optimal_solution_path)
    deviation = (best_distance - optimal_cost) / optimal_cost * 100
        

    for route in best_routes:
        for i in range(len(route)):
            route[i] += 1

    solve_time = time.time() - start_time
    print("best_routes:", best_routes)
    print("best_distance:", best_distance)
    print("optimal_cost:", optimal_cost)
    print(f"deviation: {deviation:.2f}%")
    return deviation,solve_time,len(demands)
