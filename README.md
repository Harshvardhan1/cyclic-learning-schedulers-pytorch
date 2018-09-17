# Cyclic learning rate schedulers -PyTorch

### Implementation
Cyclic learning rate schedules -
- cyclic cosine annealing - CycilcCosAnnealingLR()
- cyclic linear decay - CyclicLinearLR()

### Requirements
- numpy 
- python >= 2.7
- PyTorch >= 0.4.0

### Reference
<a href= https://arxiv.org/pdf/1608.03983.pdf> *SGDR: Stochastic Gradient Descent with Warm Restarts* </a>

### Usage
Sample - (follow similarly for CyclicLinearLR)
```
from cyclicLR import CyclicCosAnnealingLR
import torch

optimizer = torch.optim.SGD(lr=1e-3)
scheduler = CyclicCosAnnealingLR(optimizer,milestones=[30,80],eta_min=1e-6)
for epoch in range(100):
  scheduler.step()
  train(..)
  validate(..)
```
>Note: scheduler.step() shown is called at every epoch. It can be called even in every batch. Remember to specify milestones in number of batches (and not number of epochs) in such as case.


### Visualization
Cyclic Cosine Annealing Learning Rate Schedule
![Cosine LR](cosine.png)


Cyclic Linear Learning Rate Schedule
![Linear LR](linear.png)


