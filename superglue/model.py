import pdb
import torch_geometric
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
import torch
import time
from itertools import product, permutations
from superglue.sinkhorn import sinkhorn_stabilized

def generate_edges_intra(len1, len2):
    edges1 = torch.tensor(list(permutations(range(len1), 2)), dtype=torch.long, requires_grad=False).t().contiguous()
    edges2 = torch.tensor(list(permutations(range(len2), 2)), dtype=torch.long, requires_grad=False).t().contiguous() + len1
    edges = torch.cat([edges1, edges2], dim=1)
    return edges


def generate_edges_cross(len1, len2):
    edges = torch.tensor(list(product(range(len1), range(len1, len1+len2)))).t().contiguous()
    return edges

class AttConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, **kwargs):
        super(AttConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.W1 = Parameter(torch.Tensor(self.in_channels, self.heads * out_channels))
        self.W2 = Parameter(torch.Tensor(self.in_channels, self.heads * out_channels))
        self.W3 = Parameter(torch.Tensor(self.in_channels, self.heads * out_channels))

        self.bias1 = Parameter(torch.Tensor(heads * out_channels))
        self.bias2 = Parameter(torch.Tensor(heads * out_channels))
        self.bias3 = Parameter(torch.Tensor(heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W1)
        glorot(self.W2)
        glorot(self.W3)

        zeros(self.bias1)
        zeros(self.bias2)
        zeros(self.bias3)

    def forward(self, x, edge_index, size=None):

        q = torch.matmul(x, self.W1) + self.bias1
        k = torch.matmul(x, self.W2) + self.bias2
        v = torch.matmul(x, self.W3) + self.bias3

        return self.propagate(edge_index, size=None, x=x, q=q, k=k, v=v)

    def message(self, q, k, v, v_i, v_j, q_i, q_j, k_i, k_j):
        # Compute attention coefficients.
        #print(f"got {v_i.shape} {v_j.shape} {q_i.shape} {q_j.shape} {k_i.shape} {k_j.shape}")

        alpha = torch.nn.functional.softmax(q_i * k_j, dim=1)
        m = alpha * v_j
        return m

    def update(self, aggr_out):
        return aggr_out


class superglue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 32, bias=True)
        self.fc2 = torch.nn.Linear(32, 64, bias=True)

        self.mp1 = AttConv(in_channels=64, out_channels=64, heads=2)
        self.mp2 = AttConv(in_channels=128, out_channels=64, heads=2)
        self.mp3 = AttConv(in_channels=128, out_channels=64, heads=2)
        self.mp4 = AttConv(in_channels=128, out_channels=64, heads=2)

        self.fc3 = torch.nn.Linear(128, 128, bias=True)
        self.dustbin_weight = Parameter(torch.Tensor(1))

    def pos_encoder(self, p):
        x = self.fc1(p)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        return x

    def forward(self, p1, d1, p2, d2, matches):
        x1 = self.pos_encoder(p1) + d1
        x2 = self.pos_encoder(p2) + d2

        x = torch.cat([x1, x2], dim=1)[0, :, :]

        len1 = p1.shape[1]
        len2 = p2.shape[1]

        edges_intra = generate_edges_intra(len1, len2)
        edges_cross = generate_edges_cross(len1, len2)

        x1 = self.mp1.forward(x, edges_intra)
        x2 = x1 + self.mp2.forward(x1, edges_cross)
        x3 = x2 + self.mp3.forward(x2, edges_intra)
        x4 = x3 + self.mp4.forward(x3, edges_cross)

        x5 = torch.nn.functional.relu(self.fc3.forward(x4))

        #p11 = torch.nn.functional.relu(self.fc3.forward(p1))
        #p12 = torch.nn.functional.relu(self.fc3.forward(p1))
        #p11 = p11 / torch.norm(p11, dim=2, keepdim=True)
        x5 = x5 / torch.norm(x5, dim=1, keepdim=True)
        v1 = x5[:len1, :]
        v2 = x5[len1:, :]

        #pdb.set_trace()

        #v1 = p11[0, :, :]
        #v2 = p12[0, :, :]

        costs = torch.tensordot(v1, v2, dims=([1], [1]))

        dustbin_x = torch.ones((1, costs.shape[1])) * self.dustbin_weight
        dustbin_y = torch.ones((costs.shape[0] + 1, 1)) * self.dustbin_weight

        costs_x = torch.cat([costs, dustbin_x], dim=0)
        costs_with_dustbin = torch.cat([costs_x, dustbin_y], dim=1)

        costs_with_dustbin2 = 1 + (-costs_with_dustbin)  # / costs_with_dustbin.sum()
        n1 = torch.ones((len1 + 1, 1), requires_grad=False)
        n2 = torch.ones((len2 + 1, 1), requires_grad=False)

        n1[-1, 0] = len2
        n2[-1, 0] = len1

        sol = sinkhorn_stabilized(n1[:, 0], n2[:, 0], costs_with_dustbin2, reg=0.001)
        loss = []

        #print(sol)
        acc = []

        for match in matches:
            loss.append(-torch.log(sol[match[0], match[1]] + 1e-3).reshape(-1))
            acc.append((torch.argmax(sol[match[0], :]) == match[1]).reshape(-1).float())
        loss = torch.cat(loss).mean()
        acc = torch.cat(acc).mean()
        print(f"Loss: {str(loss.item())} | Acc: {acc * 100}%")
        return loss

