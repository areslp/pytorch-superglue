import torch
import numpy as np
import pdb

import torch.nn.modules
from superglue.model import superglue
from torch.autograd import Variable
from torch import optim
torch.autograd.set_detect_anomaly(True)

def generate_keypoints():
    batch_size = 1
    num_pts = 32
    num_features = 64

    p1 = np.random.randint(0, 100, size=(batch_size, num_pts, 2)).astype(np.float32)
    p2 = np.random.randint(0, 100, size=(batch_size, num_pts, 2)).astype(np.float32)
    p2 = p1

    d1 = np.random.normal(0.0, 1.0, size=(batch_size, num_pts, num_features)) + \
         np.linspace(0, 31, num_features).reshape((1, -1, num_features))

    d2 = np.random.normal(0.0, 1.0, size=(batch_size, num_pts, num_features)) + \
         np.linspace(0, 31, num_features).reshape((1, -1, num_features))

    d1 -= d1.mean(axis=2).reshape(1, -1, 1)
    d2 -= d2.mean(axis=2).reshape(1, -1, 1)

    d1 *= 0
    d2 *= 0

    p1 = (p1 - 50) / 100
    p2 = (p2 - 50) / 100

    return p1, p2, d1, d2


p1, p2, d1, d2 = generate_keypoints()

'''
pt1 = torch.from_numpy(np.random.rand(12, 2))
pt2 = pt1 + torch.from_numpy(np.random.normal(0.0, 0.01, (12, 2)))

pt1[11, 0] = 2.0
pt1[11, 1] = 2.0

pt2[11, 0] = -1.0
pt2[11, 1] = -1.0

C = torch.cdist(pt1, pt2) #torch.tensordot(pt1, pt2, dims=([1], [1]))

n1 = torch.ones((12 + 1, 1))
n2 = torch.ones((12 + 1, 1))
n1[-1] = 12
n2[-1] = 12


C2 = torch.ones(n1.shape[0], n2.shape[0]) * 0.55
C2[:12, :12] = C

solver = SinkhornSolver(epsilon=1e-6)
F = solver.forward(n1, n2, C2)
FF = F[1]
FF = FF / FF.max()
pdb.set_trace()
'''

p1 = torch.from_numpy(p1).float()
d1 = torch.from_numpy(d1).float()

p2 = torch.from_numpy(p2).float()
d2 = torch.from_numpy(d2).float()

model = superglue()
matches = []

for i in range(p1.shape[1]):
    matches.append((i, i))

#for i in range(p1.shape[1], p2.shape[1]):
#    matches.append((-1, i))

#model.forward(p1, d1, p2, d2, matches)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)


for epoch_num in range(100):

    epoch_loss = []

    for idx in range(100):
            optimizer.zero_grad()

            loss = model.forward(p1, d1, p2, d2, matches)

            if bool(loss == 0):
                continue

            loss.backward()

            #torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            #print(loss)

            optimizer.step()
