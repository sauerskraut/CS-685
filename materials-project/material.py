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

class MaterialStructureEnvironment:
    def __init__(self, material_id):
        self.actual_structure = self.initialize_structure(material_id)
        self.structure_graph = self.convert_to_graph(self.actual_structure)
        self.predicted_structure = self.initialize_predicted_structure()
        self.predicted_graph = self.convert_to_graph(self.predicted_structure)

    def initialize_structure(self, material_id = "mp-19049"):
        struct = None
        with MPRester(API_KEY) as mpr:
            struct = mpr.summary.search(material_ids = [material_id])[0].structure

        return struct

    def convert_to_graph(self, structure):
        # Convert the structure to a graph using PyTorch Geometric
        # Get the node features
        atomic_numbers = [site.specie.number for site in structure]
        node_features = torch.tensor(atomic_numbers, dtype=torch.float).view(-1, 1)

        # Use CrystalNN to find neighbors and create edge index and edge attributes
        cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0)
        edge_index = []
        edge_attr = []

        for i in range(len(structure)):
            neighbors = cnn.get_nn_info(structure, i)
            for neighbor in neighbors:
                edge_index.append([i, neighbor['site_index']])
                edge_attr.append(neighbor['weight'])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

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

    def calculate_reward(self):
        # Calculate the degree of each node in the actual and predicted graphs
        actual_degrees = degree(self.structure_graph.edge_index[0])
        predicted_degrees = degree(self.predicted_graph.edge_index[0])

        # Calculate the degree distributions
        actual_distribution = torch.bincount(actual_degrees)
        predicted_distribution = torch.bincount(predicted_degrees)

        # Pad the smaller distribution with zeros to match the size of the larger one
        if actual_distribution.size(0) > predicted_distribution.size(0):
            padding = torch.zeros(actual_distribution.size(0) - predicted_distribution.size(0), dtype=torch.long)
            predicted_distribution = torch.cat([predicted_distribution, padding])
        elif predicted_distribution.size(0) > actual_distribution.size(0):
            padding = torch.zeros(predicted_distribution.size(0) - actual_distribution.size(0), dtype=torch.long)
            actual_distribution = torch.cat([actual_distribution, padding])

        # Calculate the difference between the degree distributions
        difference = torch.abs(actual_distribution - predicted_distribution).sum()

        # The similarity measure is the inverse of the difference
        similarity = 1 / (1 + difference.item())

        # Return the similarity as the reward
        return similarity
    
if __name__ == "__main__":
    material_id = "mp-19049"
    env = MaterialStructureEnvironment(material_id)
    env.display_graph(env.structure_graph)
    env.display_graph(env.predicted_graph)
    #print(env.calculate_reward())