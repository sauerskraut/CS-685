import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import os
from datetime import datetime

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class ChebNet(torch.nn.Module):
    def __init__(self, dataset):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_node_features, 16, K=2)
        self.conv2 = ChebConv(16, dataset.num_classes, K=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, dataset):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, 16, heads=2, dropout=0.2)
        self.conv2 = GATConv(16*2, dataset.num_classes, heads=1, concat=False, dropout=0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
def train_evaluate(model, dataset_name, root_dir):
    # Load specified dataset
    dataset = Planetoid(root=root_dir, name=dataset_name)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model(dataset).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Move data object to CPU before converting to NetworkX for visualization
    data = data.to('cpu')

    # Evaluation
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))

    G = to_networkx(data, to_undirected=True)
    plt.figure(1,figsize=(14,12)) 
    nx.draw(G, cmap=plt.get_cmap('Set1'),node_color = pred.cpu().numpy(), node_size=75, linewidths=6)
    plt.text(0.05, 0.05, f'Accuracy: {acc:.4f}', horizontalalignment='left', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.suptitle(f'GCN on {dataset_name} Dataset Using Model ' + model.__class__.__name__, fontsize=16)

    # FIX ME
    # Save the plot
    # date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save_file = os.path.join(root_dir, f'{dataset_name}_{date_time}_plot.png')
    # plt.savefig(save_file)

    plt.show()

def runner(dataset, root_dir='./data'):
    train_evaluate(GCN, dataset, root_dir)
    train_evaluate(ChebNet, dataset, root_dir)
    train_evaluate(GAT, dataset, root_dir)

# Try running the Cora dataset through different models
if __name__ == '__main__':
    dataset_name = 'Cora'
    root_dir = './data'
    train_evaluate(GCN, dataset_name, root_dir)
    train_evaluate(ChebNet, dataset_name, root_dir)
    train_evaluate(GAT, dataset_name, root_dir)
