import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import torch.optim as optim
from material import MaterialStructureEnvironment, Action

class PolicyNetwork(nn.Module):
    def __init__(self, node_feature_size, edge_attr_size, num_actions):
        super(PolicyNetwork, self).__init__()
        self.node_conv = GCNConv(node_feature_size, 64)
        self.edge_conv = GCNConv(edge_attr_size, 64)
        self.fc = nn.Linear(128, num_actions)  # Add a linear layer

    def forward(self, data):
        node_x, edge_index, edge_x = data.x, data.edge_index, data.edge_attr

        node_x = torch.relu(self.node_conv(node_x, edge_index))
        edge_x = torch.relu(self.edge_conv(edge_x, edge_index))

        node_x = global_mean_pool(node_x, torch.zeros(node_x.size(0), dtype=torch.long))
        edge_x = global_mean_pool(edge_x, torch.zeros(edge_x.size(0), dtype=torch.long))

        x = torch.cat((node_x, edge_x), dim=1)
        x = self.fc(x)  # Pass the concatenated vector through the linear layer
        x = torch.softmax(x, dim=1)
        return x

# Initialize the environment
env = MaterialStructureEnvironment()

# Initialize the network and the optimizer
node_feature_size = env.structure_graph.x.shape[1]
edge_attr_size = env.structure_graph.edge_attr.shape[1]
num_actions = 6  # Number of possible actions
network = PolicyNetwork(node_feature_size, edge_attr_size, num_actions)
optimizer = optim.Adam(network.parameters(), lr=0.01)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    # Reset the environment and the episode data
    env = MaterialStructureEnvironment()
    episode_rewards = []
    episode_log_probs = []

    # Generate an episode
    done = False
    while not done:
        # Get the current state
        node_features = env.structure_graph.x
        edge_attrs = env.structure_graph.edge_attr
        edge_index = env.structure_graph.edge_index  # You also need edge_index for GCNConv

        # Create a Data object
        data = Data(x=node_features, edge_attr=edge_attrs, edge_index=edge_index)

        # Select an action
        action_probs = network(data)  # Pass the Data object to the network
        action = torch.multinomial(action_probs, 1)
        log_prob = torch.log(action_probs[action])

        # Execute the action
        _, reward = env.step(Action(action.item()))

        # Print diagnostic information
        print(f"Episode: {episode}, Action: {action.item()}, Reward: {reward}")

        # Store the reward and the log probability of the action
        episode_rewards.append(reward)
        episode_log_probs.append(log_prob)

        # Check if the episode is done
        done = env.is_done()

    # Calculate the return
    returns = []
    R = 0
    for r in episode_rewards[::-1]:
        R = r + 0.99 * R  # Discount factor
        returns.insert(0, R)
    returns = torch.tensor(returns)

    # Update the policy
    loss = -torch.sum(torch.tensor(episode_log_probs) * returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss
    print(f"Episode: {episode}, Loss: {loss.item()}")