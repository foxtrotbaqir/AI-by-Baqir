from collections import deque
from graph import my_graph


# Uses BFS to find all paths between two vertices
def bfs_all_paths(graph, start_vertex, goal_vertex):
    queue = deque()
    queue.append((start_vertex, [start_vertex]))
    while queue:
        current_vertex, current_path = queue.popleft()
        for neighbor in graph[current_vertex] - set(current_path):
            if neighbor == goal_vertex:
                # Yield a path if the goal vertex is reached
                yield current_path + [neighbor]
            else:
                queue.append((neighbor, current_path + [neighbor]))


# BFS to find the shortest path and its length
def shortest_path(graph, start, goal, max_vertices):
    try:
        paths = list(bfs_all_paths(graph, start, goal))
        # Find the path with the maximum number of vertices
        current_max_vertices = max(paths, key=len)
        # Update the external variable if the current longest path is longer
        if len(current_max_vertices) > len(max_vertices):
            max_vertices.clear()
            max_vertices.extend(current_max_vertices)
        return len(max_vertices)
    except StopIteration:
        return None


# Test usage
B = [] # an empty subset of V on shortest path
# Call the function with the graph, start, goal, and longest_path
sample_graph = my_graph()
s = int(input("Enter the start vertex: "))
t = int(input("Enter the end vertex: "))
result = shortest_path(sample_graph, s, t, B)

# Check the result, which will contain the longest path found
print("Maximum number of vertices in shortest path are", result)
