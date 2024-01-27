# Customized graph building function which produces a graph based on user inputs in python dictionary format.

def my_graph():
    graph = {}

    while True:
        try:
            vertex_str = input("Enter a vertex (or 'done' to finish): ")
            if vertex_str.lower() == 'done':
                break

            vertex = int(vertex_str)
            neighbors_str = input(
                f"Enter neighbors for vertex {vertex} (comma-separated, or 'empty' for no neighbors): ")

            if neighbors_str.lower() == 'empty':
                neighbors = set()
            else:
                neighbors = set(map(int, neighbors_str.split(',')))

            graph[vertex] = neighbors
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    return graph
