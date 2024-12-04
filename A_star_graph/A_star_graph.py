from heapq import heappop, heappush
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def a_star_search(graph: dict, start: str, goal: str, heuristic_values: dict):
   
    # The open list is a priority queue of nodes to be evaluated, ordered by their f(x) value.
    # The closed list is a set of nodes that have already been evaluated.
    open_list, closed_list = [(heuristic_values[start], start)], set()
    came_from = {}  # For reconstructing the path
    steps = []  # For animation steps
    while open_list:
        cost, node = heappop(open_list)
        steps.append((node, "visited"))  # Add current node as visited for animation

        # The algorithm end when reaching the goal node,
        # NOT when it is added to the open list.
        if node == goal:
           
            path = [] # Reconstruct the path from the goal node to the start node
            while node: 
                path.append(node)
                node = came_from.get(node)
            steps.append((list(reversed(path)), "path"))  # Final path for animation
            return cost, list(reversed(path)), steps

        if node in closed_list:
            continue

        closed_list.add(node)

        # Subtract the heuristic value as it was overcounted.
        cost -= heuristic_values[node]# Subtract the heuristic value as it was overcounted.

        for neighbor, edge_cost in graph[node]:
            if neighbor in closed_list:
                continue

            '''             
            f(x) = g(x) + h(x)
            ''' 
            neighbor_cost = cost + edge_cost + heuristic_values[neighbor]
            heappush(open_list, (neighbor_cost, neighbor))
            came_from[neighbor] = node
            steps.append((neighbor, "open"))  # Mark neighbor as open for animation

    return -1, [], steps  # No path found

def animate_graph(graph, heuristic_values, steps):
    # Create a directed graph using NetworkX
    G = nx.DiGraph()
    for node, neighbors in graph.items():
        for neighbor, cost in neighbors:
            G.add_edge(node, neighbor, weight=cost)

    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Use spring layout for better spacing
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw initial graph
    nx.draw(G, pos, with_labels=True, node_size=800, node_color='lightblue', font_weight='bold')
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)},
        font_color='red',
        font_size=8
    )

    # Animation variables
    visited_nodes = set()
    open_nodes = set()
    path_nodes = set()
    path_edges = []

    def update(frame):
        nonlocal visited_nodes, open_nodes, path_nodes, path_edges

        node, action = steps[frame]

        if action == "visited":
            visited_nodes.add(node)
        elif action == "open":
            open_nodes.add(node)
        elif action == "path":
            path_nodes = set(node)
            path_edges = list(zip(node[:-1], node[1:]))

        # Clear the plot
        ax.clear()
        nx.draw(G, pos, with_labels=True, node_size=800, node_color='lightblue', font_weight='bold')
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)},
            font_color='red',
            font_size=8
        )

        # Highlight visited nodes
        nx.draw_networkx_nodes(G, pos, nodelist=list(visited_nodes), node_color='yellow', node_size=800)
        # Highlight open nodes
        nx.draw_networkx_nodes(G, pos, nodelist=list(open_nodes), node_color='orange', node_size=800)
        # Highlight path nodes and edges
        nx.draw_networkx_nodes(G, pos, nodelist=list(path_nodes), node_color='green', node_size=900)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='green', width=3)

        ax.set_title("A* Search Animation", fontsize=16)

    ani = FuncAnimation(fig, update, frames=len(steps), interval=1000, repeat=False)
    plt.show()

def main(): 
    EXAMPLE_GRAPH = {
        'S': [('A', 3), ('B', 6), ('C', 7)],
        'A': [('B', 2), ('D', 4)],
        'B': [('D', 6)],
        'C': [('D', 5), ('E', 8), ('F', 3)],
        'D': [('F', 1), ('I', 7), ('H', 6)],
        'E': [('G', 9)],
        'F': [('G', 5)],
        'H': [('J', 2), ('I', 1)],
        'I': [('K', 4), ('G', 3), ('J', 2)],
        'J': [('K', 3)],
        'K': [('G', 6)]
    }

    # Node heuristic values for the A* algorithm
    EXAMPLE_HEURISTIC_VALUES = {
        'S': 7,
        'A': 8,
        'B': 6,
        'C': 5,
        'D': 5,
        'E': 3,
        'F': 3,
        'G': 0,
        'H': 7,
        'I': 4,
        'J': 5,
        'K': 3
    }

    cost, path, steps = a_star_search(EXAMPLE_GRAPH, 'S', 'G', EXAMPLE_HEURISTIC_VALUES) # Start node: S, Goal node: G
    print(f"Path Cost: {cost}, Path: {path}")
    # Visualize the graph and path
    # Animate the search
    animate_graph(EXAMPLE_GRAPH, EXAMPLE_HEURISTIC_VALUES, steps)
if __name__ == "__main__":
    main()