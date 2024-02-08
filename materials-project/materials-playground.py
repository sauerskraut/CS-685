from mp_api.client import MPRester
from config import API_KEY
from torch_geometric.data import Data
import torch
from pymatgen.analysis.local_env import CrystalNN
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from sklearn.manifold import MDS

with MPRester(API_KEY) as mpr:
    docs = mpr.summary.search(material_ids = ["mp-19049"])

    # Get the structure
    structure = docs[0].structure

    print(structure)

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

    # After creating the edge_index, edge_attr, and node_features
    print("Edge Index:", edge_index)
    print("Edge Attributes:", edge_attr)
    print("Node Features:", node_features)

    # Create the PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    print(data)

    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # After creating the Data object
    print("Data Object:", data)

    # After converting to a NetworkX graph
    print("NetworkX Graph:", G)

    # # # Create the layout
    # pos = nx.spring_layout(G)

    # # Draw the graph
    # nx.draw(G, pos, with_labels=True, node_color=node_features.numpy(), cmap=plt.get_cmap('Set1'))

    # Try a different layout
    # pos = nx.shell_layout(G)

    # # Try plotting without the node colors
    # nx.draw(G, pos, with_labels=True)

    # plt.show()

    # Get the shortest path length between all pairs of nodes
    distances = nx.floyd_warshall_numpy(G)

    # Use MDS to find 2D positions that preserve these distances
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
    pos = mds.fit_transform(distances)

    # Create a dictionary that maps each node to its position
    pos_dict = {node: pos[i] for i, node in enumerate(G.nodes)}

    # Draw the graph
    nx.draw(G, pos=pos_dict, with_labels=True)
    plt.show()