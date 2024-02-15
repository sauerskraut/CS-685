from mp_api.client import MPRester
from config import API_KEY
from torch_geometric.data import Data
import torch
from pymatgen.analysis.local_env import CrystalNN
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from sklearn.manifold import MDS
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from networkx.algorithms.similarity import graph_edit_distance
from torch_geometric.utils import degree

class Action:
    def __init__(self, action_type, node_id1=None, node_id2=None, attributes=None):
        self.type = action_type
        self.node_id1 = node_id1
        self.node_id2 = node_id2
        self.attributes = attributes

class MaterialStructureEnvironment:
    def __init__(self, material_id = "mp-19049"):
        self.actual_structure = self.initialize_structure(material_id)
        self.structure_graph = self.convert_to_graph(self.actual_structure)
        self.predicted_structure = self.initialize_predicted_structure()
        self.predicted_graph = self.convert_to_graph(self.predicted_structure)

    def initialize_structure(self, material_id):
        struct = None
        with MPRester(API_KEY) as mpr:
            struct = mpr.summary.search(material_ids = [material_id])[0].structure

        return struct

    def convert_to_graph(self, structure):
        # Convert the structure to a graph using PyTorch Geometric
        # Get the node features
        atomic_numbers = [site.specie.number for site in structure]
        atomic_radii = [site.specie.atomic_radius for site in structure]
        node_features = torch.tensor([atomic_numbers, atomic_radii], dtype=torch.float).t()

        # Use CrystalNN to find neighbors and create edge index and edge attributes
        cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0)
        edge_index = []
        edge_attr = []

        for i in range(len(structure)):
            neighbors = cnn.get_nn_info(structure, i)
            for neighbor in neighbors:
                # Calculate the distance between the current atom and its neighbor
                site_distance = structure[i].distance(structure[neighbor['site_index']])
                edge_index.append([i, neighbor['site_index']])
                edge_attr.append([neighbor['weight'], site_distance])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 2)

        # Create the PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        return data
    
    def initialize_predicted_structure(self):
        # Start with a copy of the actual structure
        predicted_structure = self.actual_structure.copy()

        # Convert the structure to a graph
        data = self.convert_to_graph(predicted_structure)
        G = to_networkx(data, to_undirected=True)

        # Find the leaf nodes (nodes with only one connection)
        leaf_nodes = [node for node, degree in G.degree() if degree == 1]

        # Find the nodes with high atomic numbers
        high_atomic_number_nodes = [i for i, node_feature in enumerate(data.x.tolist()) if node_feature[0] > 20]

        # Combine the lists of nodes to remove
        nodes_to_remove = leaf_nodes + high_atomic_number_nodes

        # Shuffle the list of nodes to remove and select a subset
        random.shuffle(nodes_to_remove)
        nodes_to_remove = nodes_to_remove[:5]  # Change this number to control how many nodes are removed

        # Remove the nodes from the predicted structure
        predicted_structure.remove_sites(nodes_to_remove)

        return predicted_structure
        
    def display_graph(self, data):
        # Convert the PyTorch Geometric Data object to a NetworkX graph
        G = to_networkx(data, to_undirected=True)

        # Draw the graph
        nx.draw(G, with_labels=True)
        plt.show()

    def step(self, action):
        # Apply the action to the predicted structure
        self.predicted_structure = self.apply_action(action)

        # Calculate the reward
        reward = self.calculate_reward()

        return self.predicted_structure, reward

    def apply_action(self, action):
        # Apply the action to the predicted structure
        return self.predicted_structure

    def calculate_reward(self):# Calculate the degree of each node in the actual and predicted graphs
        # Get the sets of nodes in the actual and predicted graphs
        actual_nodes = set(self.structure_graph.edge_index[0].numpy())
        predicted_nodes = set(self.predicted_graph.edge_index[0].numpy())

        # Calculate the Jaccard similarity coefficient
        intersection = len(actual_nodes & predicted_nodes)
        union = len(actual_nodes | predicted_nodes)
        similarity = intersection / union

        # Convert the similarity to a percentage
        percentage = similarity * 100

        # Return the similarity as the reward
        return percentage
    
    def add_node(self, attributes):
        # Add a new node to the predicted structure with the given attributes
        self.predicted_structure.append(attributes)

    def remove_node(self, node_id):
        # Remove the node with the given ID from the predicted structure
        del self.predicted_structure[node_id]

    def add_edge(self, node_id1, node_id2, attributes):
        # Add a new edge between the nodes with the given IDs to the predicted structure
        self.predicted_structure.add_edge(node_id1, node_id2, attributes)

    def remove_edge(self, node_id1, node_id2):
        # Remove the edge between the nodes with the given IDs from the predicted structure
        self.predicted_structure.remove_edge(node_id1, node_id2)

    def modify_node(self, node_id, new_attributes):
        # Modify the attributes of the node with the given ID in the predicted structure
        self.predicted_structure[node_id] = new_attributes

    def modify_edge(self, node_id1, node_id2, new_attributes):
        # Modify the attributes of the edge between the nodes with the given IDs in the predicted structure
        self.predicted_structure[node_id1][node_id2] = new_attributes

    def apply_action(self, action):
        # Apply the action to the predicted structure
        if action.type == 0:
            # add node
            self.add_node(action.attributes)
        elif action.type == 1:
            # remove node
            self.remove_node(action.node_id)
        elif action.type == 2:
            # add edge
            self.add_edge(action.node_id1, action.node_id2, action.attributes)
        elif action.type == 3:
            # remove edge
            self.remove_edge(action.node_id1, action.node_id2)
        elif action.type == 4:
            # modify node
            self.modify_node(action.node_id, action.new_attributes)
        elif action.type == 5:
            # modify edge
            self.modify_edge(action.node_id1, action.node_id2, action.new_attributes)

        # Convert the modified structure to a graph
        self.predicted_graph = self.convert_to_graph(self.predicted_structure)

        return self.predicted_structure
    
if __name__ == "__main__":
    material_id = "mp-19049"
    env = MaterialStructureEnvironment(material_id)
    env.display_graph(env.structure_graph)
    env.display_graph(env.predicted_graph)
    print(env.calculate_reward())