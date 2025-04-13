import pickle
import pathlib

class GraphMemory:
    def __init__(self, storage: str = ""):
        # `nodes` maps an encoded state (string) to the actual state (could be a matrix or any representation)
        self.nodes = {}
        # `edges` maps an encoded state to a list of transitions (edges) represented as dictionaries.
        # Each edge contains the action taken, the next state's encoded representation, and optional metadata.
        self.edges = {}
        
        self.storage = storage
        
        if self.storage and pathlib.Path(self.storage).exists():
            with open(storage, 'rb') as f:
                self.nodes, self.edges = pickle.load(f)

    def encode_state(self, state):
        """
        Encodes the state into a unique string identifier.
        The state can be a string (already encoded) or a 2D list/tuple.
        Adjust this function if you have a different state structure.
        """
        if isinstance(state, str):
            return state
        elif isinstance(state, (list, tuple)):
            # Assuming the state is a 2D list or tuple (like a TicTacToe board),
            # we flatten it into a single string.
            return ''.join(''.join(map(str, row)) for row in state)
        else:
            return str(state)

    def add_node(self, state):
        """
        Adds a node corresponding to the state if it doesn't exist.
        Returns the encoded representation of the state.
        """
        encoded = self.encode_state(state)
        if encoded not in self.nodes:
            self.nodes[encoded] = state
            self.edges[encoded] = []  # Initialize the edge list for this node.
        return encoded

    def store(self, state, action, next_state, metadata=None):
        """
        Stores a memory entry by adding nodes for the current state and next state,
        then recording an edge from current state to next state with the action label.
        
        Args:
            state: The current state (e.g., a TicTacToe board).
            action: The action taken (e.g., position placed, move label, etc.).
            next_state: The state after the action has been applied.
            metadata: Optional dictionary with additional details (e.g., reward, timestamp).
        """
        if metadata is None:
            metadata = {}

        # Encode and add nodes to the graph.
        encoded_current = self.add_node(state)
        encoded_next = self.add_node(next_state)

        # Create an edge with the associated action and metadata.
        edge = {
            'action': action,
            'next_state': encoded_next,
            'metadata': metadata
        }
        self.edges[encoded_current].append(edge)

    def retrieve(self, state):
        """
        Retrieves all edges (i.e., stored transitions) from the given state.
        
        Args:
            state: The state whose outgoing transitions are to be retrieved.
        
        Returns:
            A list of edges from the node corresponding to the provided state.
            Each edge is a dictionary with keys: action, next_state, and metadata.
        """
        encoded = self.encode_state(state)
        return self.edges.get(encoded, [])

    def display_graph(self):
        """
        Helper method to visualize the graph contents.
        Outputs each node and its outgoing transitions.
        """
        for state_id in self.nodes:
            print(f"State: {state_id}")
            for edge in self.edges[state_id]:
                print(f"  --[{edge['action']}]--> {edge['next_state']} (metadata: {edge['metadata']})")

    def export_graph(self):
        if self.storage:
            with open(self.storage, 'wb') as f:
                pickle.dump((self.nodes, self.edges), f)

# --- Example Usage ---

if __name__ == "__main__":
    # Create instance of GraphMemory
    graph_memory = GraphMemory()

    # Example states for a tic-tac-toe game:
    state1 = [
        ['X', 'O', 'X'],
        [' ', 'O', ' '],
        [' ', ' ', 'X']
    ]
    # Suppose the action is placing an 'O' in the bottom left corner.
    action = "place O at (3,1)"

    # The resulting state after the action:
    state2 = [
        ['X', 'O', 'X'],
        [' ', 'O', ' '],
        ['O', ' ', 'X']
    ]

    # Store the transition in the graph memory.
    graph_memory.store(state1, action, state2, metadata={'reward': 0.5})

    # Retrieve transitions for state1.
    transitions = graph_memory.retrieve(state1)
    print("Transitions from state1:")
    for trans in transitions:
        print(trans)

    # Optionally, display the whole graph.
    print("\nGraph Structure:")
    graph_memory.display_graph()
