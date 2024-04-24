#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h> // OpenMP library for parallel processing

using namespace std;

// Class representing a graph
class Graph
{
    int V;                       // Number of vertices
    vector<vector<int>> adjList; // Adjacency list representation

public:
    // Constructor
    Graph(int V)
    {
        this->V = V;
        adjList.resize(V);
    }

    // Function to add an edge between two vertices
    void addEdge(int src, int dest)
    {
        adjList[src].push_back(dest);
        adjList[dest].push_back(src); // For undirected graph
    }

    // Function to get neighbors of a vertex
    vector<int> getNeighbors(int vertex)
    {
        return adjList[vertex];
    }
};

// Function for parallel Breadth First Search (BFS)
void parallelBFS(Graph &graph, int source, vector<bool> &visited)
{
    queue<int> q;
    q.push(source);
    visited[source] = true;

    while (!q.empty())
    {
        int current = q.front();
        q.pop();
        cout << "Visited: " << current << endl;

        // Get neighbors of the current vertex
        vector<int> neighbors = graph.getNeighbors(current);

// Parallel loop over neighbors to visit them
#pragma omp parallel for
        for (int i = 0; i < neighbors.size(); ++i)
        {
            int neighbor = neighbors[i];
            // Check if the neighbor has been visited
            if (!visited[neighbor])
            {
                // Mark neighbor as visited and enqueue it
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

// Function for parallel Depth First Search (DFS)
void parallelDFS(Graph &graph, int source, vector<bool> &visited)
{
    stack<int> s;
    s.push(source);
    visited[source] = true;

    while (!s.empty())
    {
        int current = s.top();
        s.pop();
        cout << "Visited: " << current << endl;

        // Get neighbors of the current vertex
        vector<int> neighbors = graph.getNeighbors(current);

// Parallel loop over neighbors to visit them
#pragma omp parallel for
        for (int i = 0; i < neighbors.size(); ++i)
        {
            int neighbor = neighbors[i];
            // Check if the neighbor has been visited
            if (!visited[neighbor])
            {
                // Mark neighbor as visited and push it onto the stack
                visited[neighbor] = true;
                s.push(neighbor);
            }
        }
    }
}

int main()
{
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    Graph graph(V);
    cout << "Enter the number of edges: ";
    cin >> E;
    cout << "Enter the edges (src dest):" << endl;
    for (int i = 0; i < E; ++i)
    {
        int src, dest;
        cin >> src >> dest;
        graph.addEdge(src, dest);
    }

    vector<bool> visited(V, false); // Array to mark visited vertices

    cout << "Parallel BFS:" << endl;
// Parallel region for BFS
#pragma omp parallel num_threads(2)
    {
#pragma omp single nowait
        parallelBFS(graph, 0, visited); // Call parallel BFS from a single thread
    }

    // Reset visited array for DFS
    fill(visited.begin(), visited.end(), false);

    cout << "Parallel DFS:" << endl;
// Parallel region for DFS
#pragma omp parallel num_threads(2)
    {
#pragma omp single nowait
        parallelDFS(graph, 0, visited); // Call parallel DFS from a single thread
    }

    return 0;
}
