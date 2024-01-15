import numpy as np
import heapq
class Graph:
    def __init__(self, graph, reliability, capacity, delay):
        self.graph = graph
        self.reliability = reliability
        self.capacity = capacity
        self.delay = delay

    def bellman_ford(self, start, end, bandwidth_demand, delay_threshold, reliability_threshold):
        num_vertices = len(self.graph)
        distance = [float('inf')] * num_vertices
        reliability_so_far = [0.0] * num_vertices
        capacity_so_far = [float('inf')] * num_vertices
        distance[start] = 0
        reliability_so_far[start] = 1.0
        capacity_so_far[start] = float('inf')

        for _ in range(num_vertices - 1):
            for u in range(num_vertices):
                for v in range(num_vertices):
                    if self.graph[u][v] != 0 and distance[u] + self.delay[u][v] <= delay_threshold \
                            and reliability_so_far[u] * self.reliability[u][v] >= reliability_threshold \
                            and self.capacity[u][v] >= bandwidth_demand:
                        if distance[u] + self.delay[u][v] < distance[v]:
                            distance[v] = distance[u] + self.delay[u][v]
                            reliability_so_far[v] = reliability_so_far[u] * self.reliability[u][v]
                            capacity_so_far[v] = min(capacity_so_far[u], self.capacity[u][v])

        return distance[end], reliability_so_far[end], capacity_so_far[end]

    def astar(self, start, end, bandwidth_demand, delay_threshold, reliability_threshold):
        num_vertices = len(self.graph)
        heap = [(0, start, [])]  # Modified heap to include the path
        visited = set()
        distance = [float('inf')] * num_vertices
        reliability_so_far = [0.0] * num_vertices
        capacity_so_far = [float('inf')] * num_vertices
        distance[start] = 0
        reliability_so_far[start] = 1.0
        capacity_so_far[start] = float('inf')

        while heap:
            current_distance, current_vertex, path_so_far = heapq.heappop(heap)
            visited.add(current_vertex)

            for v in range(num_vertices):
                if self.graph[current_vertex][v] != 0 and v not in visited \
                        and current_distance + self.delay[current_vertex][v] <= delay_threshold \
                        and reliability_so_far[current_vertex] * self.reliability[current_vertex][v] >= reliability_threshold \
                        and self.capacity[current_vertex][v] >= bandwidth_demand:
                    new_distance = current_distance + self.delay[current_vertex][v]
                    if new_distance < distance[v]:
                        distance[v] = new_distance
                        reliability_so_far[v] = reliability_so_far[current_vertex] * self.reliability[current_vertex][v]
                        capacity_so_far[v] = min(capacity_so_far[current_vertex], self.capacity[current_vertex][v])
                        heapq.heappush(heap, (new_distance, v, path_so_far + [v]))

        return distance[end], reliability_so_far[end], capacity_so_far[end], path_so_far + [end]

    def dijkstra(self, start, end, bandwidth_demand, delay_threshold, reliability_threshold):
        num_vertices = len(self.graph)
        heap = [(0, start, [])]  # Modified heap to include the path
        visited = set()
        distance = [float('inf')] * num_vertices
        reliability_so_far = [0.0] * num_vertices
        capacity_so_far = [float('inf')] * num_vertices
        distance[start] = 0
        reliability_so_far[start] = 1.0
        capacity_so_far[start] = float('inf')

        while heap:
            current_distance, current_vertex, path_so_far = heapq.heappop(heap)
            visited.add(current_vertex)

            for v in range(num_vertices):
                if self.graph[current_vertex][v] != 0 and v not in visited \
                        and current_distance + self.delay[current_vertex][v] <= delay_threshold \
                        and reliability_so_far[current_vertex] * self.reliability[current_vertex][v] >= reliability_threshold \
                        and self.capacity[current_vertex][v] >= bandwidth_demand:
                    new_distance = current_distance + self.delay[current_vertex][v]
                    if new_distance < distance[v]:
                        distance[v] = new_distance
                        reliability_so_far[v] = reliability_so_far[current_vertex] * self.reliability[current_vertex][v]
                        capacity_so_far[v] = min(capacity_so_far[current_vertex], self.capacity[current_vertex][v])
                        heapq.heappush(heap, (new_distance, v, path_so_far + [v]))

        return distance[end], reliability_so_far[end], capacity_so_far[end], path_so_far + [end]

graph = [ 
    [0,3,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,0,3,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,3,0,5,2,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,5,0,5,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,2,5,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,5,0,0,0,0,3,0,3,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,5,4,0,3,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,4,0,1,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,3,2,0,0,2,3,3,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,2,2,0,0,0,5,4,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,2,0,0,3,0,0,1,0,0,4,0,0,0,2,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,3,0,1,0,3,0,0,2,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,5,0,3,0,2,0,0,3,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,4,0,0,2,0,0,0,0,5,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,3,0,0,0,4,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,2,0,0,3,0,4,0,0,0,5,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,4,0,4,0,0,0,1,4,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,4,0,0,0,0,0,0,4],
    [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,3,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,1,0,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,0,2,0,3,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,3,0,3],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,3,0]
]

reliability = [
    [0.0,0.97,0.0,0.0,0.0,0.98,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.97,0.0,0.96,0.0,0.0,0.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.96,0.0,0.98,0.95,0.0,0.97,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.98,0.0,0.95,0.0,0.98,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.95,0.95,0.0,0.0,0.0,0.95,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.98,0.99,0.0,0.0,0.0,0.0,0.96,0.0,0.99,0.0,0.98,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.97,0.98,0.0,0.96,0.0,0.99,0.96,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.95,0.0,0.99,0.0,0.0,0.96,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.99,0.96,0.0,0.0,0.97,0.95,0.98,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.96,0.97,0.0,0.0,0.0,0.98,0.95,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.98,0.0,0.0,0.95,0.0,0.0,0.97,0.0,0.0,0.97,0.0,0.0,0.0,0.95,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.98,0.0,0.97,0.0,0.96,0.0,0.0,0.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.98,0.0,0.96,0.0,0.97,0.0,0.0,0.96,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.95,0.0,0.0,0.97,0.0,0.0,0.0,0.0,0.98,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.97,0.0,0.0,0.0,0.0,0.96,0.0,0.0,0.0,0.97,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.99,0.0,0.0,0.96,0.0,0.98,0.0,0.0,0.0,0.95,0.98,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.96,0.0,0.0,0.98,0.0,0.98,0.0,0.0,0.0,0.95,0.95,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.98,0.0,0.0,0.98,0.0,0.0,0.0,0.0,0.0,0.0,0.96],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.95,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.96,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.97,0.0,0.0,0.0,0.96,0.0,0.96,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.95,0.0,0.0,0.0,0.96,0.0,0.96,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.98,0.95,0.0,0.0,0.0,0.96,0.0,0.97,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.95,0.0,0.0,0.0,0.0,0.97,0.0,0.95],
[ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.96,0.0,0.0,0.0,0.0,0.95,0.0]
]

capacity = [
    [0,10,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[10,0,7,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,7,0,5,7,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,5,0,5,0,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,7,5,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[4,3,0,0,0,0,10,0,3,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,4,10,0,10,0,9,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,6,0,9,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,3,9,0,0,9,8,10,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,9,9,0,0,0,8,4,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,8,0,0,8,0,0,8,0,0,5,0,0,0,9,0,0,0,0,0],
[0,0,0,0,0,0,0,0,10,0,8,0,7,0,0,4,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,8,0,7,0,10,0,0,8,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,4,0,0,10,0,0,0,0,3,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,9,0,0,0,4,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,4,0,0,9,0,5,0,0,0,10,10,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,5,0,10,0,0,0,10,3,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,10,0,0,0,0,0,0,5],
[0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,0,5,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,5,0,4,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0,0,0,4,0,9,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,10,0,0,0,9,0,10,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,10,0,6],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,6,0]
]

delay = [
    [0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[2,0,4,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,4,0,3,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,3,0,3,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,2,3,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[2,1,0,0,0,0,2,0,1,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,1,4,0,2,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,4,0,5,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,2,0,0,2,5,1,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,3,2,0,0,0,4,1,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,4,0,0,5,0,0,2,0,0,4,0,0,0,3,0,0,0,0,0],
[0,0,0,0,0,0,0,0,1,0,2,0,5,0,0,3,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,4,0,5,0,1,0,0,5,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,4,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,3,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,3,0,0,3,0,5,0,0,0,2,2,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,5,0,1,0,0,0,3,1,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,1,0,0,0,0,0,0,2],
[0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,4,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,4,0,1,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,1,0,5,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,0,0,0,5,0,4,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,4,0,2],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,2,0]

]

graph = Graph(graph, reliability, capacity, delay)

start_vertex = int(input("Enter the start vertex: "))
end_vertex = int(input("Enter the end vertex: "))

bandwidth_demand = 5
delay_threshold = 40
reliability_threshold = 0.7

# Bellman-Ford
bellman_ford_result = graph.bellman_ford(start_vertex, end_vertex, bandwidth_demand, delay_threshold, reliability_threshold)
print("Bellman-Ford Result:")
print("  Distance:", bellman_ford_result[0])
print("  Reliability:", bellman_ford_result[1])
print("  Capacity:", bellman_ford_result[2])
print()

# A*
astar_result = graph.astar(start_vertex, end_vertex, bandwidth_demand, delay_threshold, reliability_threshold)
print("A* Result:")
print("  Distance:", astar_result[0])
print("  Reliability:", astar_result[1])
print("  Capacity:", astar_result[2])
print("  Path:", astar_result[3])
print()

# Dijkstra
dijkstra_result = graph.dijkstra(start_vertex, end_vertex, bandwidth_demand, delay_threshold, reliability_threshold)
print("Dijkstra Result:")
print("  Distance:", dijkstra_result[0])
print("  Reliability:", dijkstra_result[1])
print("  Capacity:", dijkstra_result[2])
print("  Path:", dijkstra_result[3])