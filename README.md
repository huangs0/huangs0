# EGUM
EGUM is a simple, powerful `python` tools help you accelerate graph neural network inference, it's totally open source so you can customize it for your convinience. To use it, you need to have these dependencies:

* linux
* Python 3.0 +
* torch: now `EGUM` only support `pytorch` as computation backend
* dgl
* numpy
* ogb if you want to use example

If you want to build from source code, you will additionally need:
* gcc: to build on linux
* nvcc: to build CUDA code
* pybind11: to bind C++ code with Python

However, if you don't want to use the accelerated mode, you can simply download and put the EGUM.py in your working directory and import it just-in-time.

## Get Started
### Installation
EGUM will be published to python package manager `pip` soon, now it only support install from source code

### Tutorial
Four steps to use EGUM for GNN acceleration:
0. Train your model
1. Use Initializer to step up your dataset
2. Extend the EGUM model
3. Create instance of your customized EGUM model and run it

#### 1. Train your model
Please note that EGUM is only a GNN model inference acceleration tools, but not a training framework. To use this inference acceleration tools, you need to train your model before. Currently EGUM only support models from `dgl` with `pytorch` as backend, they're dependencies of EGUM. You can find how to create and train models from dgl documentation: https://docs.dgl.ai/guide/index.html, but two small demo about how to create classical GAT and GCN model about node classification using Cora dataset is as follows:

First import necessary packages:
```python
import torch.nn as nn
import torch
import dgl
import numpy as np
```

Prepare Cora datasets, downloading might takes a while:
```python
from dgl.data import CoraGraphDataset

data = CoraGraphDataset()

g = data[0]

features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
```

Creating our GAT models, assuming two layers:
```python
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_head=1):
        super(GAT, self).__init__()
        self.gatconv1 = GATConv(in_dim, hidden_dim, num_head)
        self.gatconv2 = GATConv(hidden_dim, out_dim, num_head)
        
    def forward(self, graph, feat):
        feat = self.gatconv1(graph, feat)
        feat = F.relu(feat)
        feat = self.gatconv2(graph, feat)
        return feat
```

Creating our GCN models, assuming two layers:
```python
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.gcnconv1 = GraphConv(in_dim, hidden_dim)
        self.gcnconv2 = GraphConv(hidden_dim, out_dim)
        
    def forward(self, graph, feat):
        feat = self.gcnconv1(graph, feat)
        feat = F.relu(feat)
        feat = self.gcnconv2(graph, feat)
        return feat
```

Then train our models:
```python
import time

# Select GCN or GAT as you want
# net = GAT(in_dim=1433, hidden_dim=600, out_dim=7)
net = GCN(in_dim=1433, hidden_dim=600, out_dim=7)

# You can use other optimizer like SGD
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# main loop
dur = []
losses = []
for epoch in range(3):
    if epoch >= 3:
        t0 = time.time()

    logits = net(g, features).squeeze()
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    losses.append(loss.item())
    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), np.mean(dur)))
```

Please note that for GNN model it cannot be trainned for too much epoch (recommend to train for 5 epoch) as it has the property of similarity.

#### 2. Initializer
Initializer is a class of EGUM to help you prepare the input of EGUM. For modifications on graph, it can be divided into six part:
1. Add node(s) to graph
2. Add edge(s) to graph
3. Modify some node(s)' features
4. Modify some edge(s)' features
5. Delete node(s) from graph
6. Delete edge(s) from graph

For these six kinds of change, EGUM provide corresponding APIs to help you setup environments, please note all following function is static and it's packaged in class `Initializer` so you can directly use `from EGUM.Initializer import xxx` to import corresponding setup functions you need instead of import the whole class.

|Type of modification|Corresponding Initializer API|Description|
|--- |--- |--- |
|Add node to graph|new_node()|Add a new node (Adding several nodes a time is still in process)|
|Add edges to graph|new_edges()|Add some new edges (Can add multiple edges a time)|
|Modify some ndoe features|change_ndoes()|Change soem nodes' features (Can change multiple node features a time)|
|Modify some edge features|Change_edges()|Change some edges' features (Can change multiple edge features a time)|
|Delete ndoes from graph|delete_nodes()|Delete some nodes from graph (Can delete multiple nodes a time)|
|Delete edges from graph|delete_edges()|Delete some edges from graph (Can delete multiple edges a time)|

Please note that these APIs may have different input type (please refer to API reference), according to different functions they perform, but they have a uniform return type:
```Python
# If no edge feature is passed in as parameters
subgraph, subgraph_node_feat, graph, graph_node_feat = initializer_functions(...)
# If pass in edge features as parameters
subgraph, subgraph_node_feat, subgraph_edge_feat, graph, graph_node_feat, graph_edge_feat = initializer_functions(...)
```
#### 3. Extending EGUM
EGUM is more like an interface with some supporting functions rather than a class, you need to extend it in order to make your model accelerated.

However, unlike usual pytorch module, you don't need to create instance of different pytorch `nn.Module` again. What you only need to do is: Pass the trained model from step 0 to the super initializer. Following is the standardrized EGUM `__init__(self, model)` function:

```python
# You need to use EGUM as the father class
class MyEGUM(EGUM):
    # Your initializer must have model as parameter
    # model is the model you train in step 0
    def __init__(self, model):
        # pass model to the super initializer
        super(MyEGUM, self).__init__(model)
        # You don't need to redefine your layers again
        # You can define some extra parameters here
```

When extending EGUM, please kindly note that you don't need to rewrite your layers again, and it's strictly forbidden to redo layer definition when extending EGUM. Instead, EGUM will dynamically load the layers you define in trained model **with parameters**. If you redefine the layers, those layer won't hold parameters from trained model.

Congratulation! Now you have complete half of extension, only thing left is to extend the `forward(self, **kwargs)` function to meet the requirement of `pytorch`. Generally speaking, all you need to do is copy the `forward()` function from your original model, paste the `forward` in new class and do three modification:
1. Insert `self.initialize(graph, node_feat, edge_feat)` at the **first line** of your code, note that if you don't have edge_feat, you can ignore this parameter
2. Insert `graph, node_feat = self.extend(graph, node_feat, edge_feat)` before **every layer** with message-passing, which means all the layers from `dgl.nn`, including `GATConv`, `GraphConv`, or other your self-defined layers. 
3. Insert `graph, node_feat = self.back(graph, node_feat, edge_feat)` before your return, if you don't use edge_feat, you can ignore this parameters.

For example, if I want to transform the `forward` from `GAT` model we create before to `MyEGUM`:

Original GAT Code:
```python
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_head=1):
        super(GAT, self).__init__()
        self.gatconv1 = GATConv(in_dim, hidden_dim, num_head)
        self.gatconv2 = GATConv(hidden_dim, out_dim, num_head)
        
    def forward(self, graph, feat):
        feat = self.gatconv1(graph, feat)
        feat = F.relu(feat)
        feat = self.gatconv2(graph, feat)
        return feat
```

GAT in MyEGUM:
```python
class MyEGUM(EGUM):
    def __init__(self, model):
        super(MyEGUM, self).__init__(model)
        
    # Extend the forward funciton
    def forward(self, graph, feat):
        # Add self.initialize at the first line, ignore the edge_feat parameter
        self.initialize(graph, feat)
        
        # Add graph, feat = self.extend(graph, feat) before every layer invoktion 
        graph, feat = self.extend(graph, feat)
        feat = self.gatconv1(graph, feat)
        feat = F.relu(feat)
        
        # Add graph, feat = self.extend(graph, feat) before every layer invoktion 
        graph, feat = self.extend(graph, feat)
        feat = self.gatconv2(graph, feat)
        
        # Add graph, feat = self.back(graph, feat) before your return
        graph, feat = self.back(graph, feat)
        return feat
```

#### 4. Create instance of your customized EGUM model and run it
Finally you can create your EGUM model and run it to see the result. To create an instance of EGUM, please note that you shall pass in the trained model from step 0 as parameter, which has been mentioned in step 3.

```python
my_egum = MyEGUM(model=net)
```

There's only a few steps to run the EGUM:
1. Prepare the feature maps for nodes and edges
2. load the feature maps and features
3. run the model

##### 4.1. Prepare feature maps
EGUM relies on the graph locality and incremental subgraph extraction to accelerate graph inference. To make use of incremental subgraph extraction, we need to extract graph features for each layers, i.e, we need to prepare feature map of following graph in each layer. In later version of EGUM, it's now only at 0.0.1, an automatic execution of graph will be provided. But now, we need to prepare the feature maps ourselves, sorry for that. For example, if we want to prepare feature maps for `GCN` model we train before, we shall:

Original Code of GCN
```python
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.gcnconv1 = GraphConv(in_dim, hidden_dim)
        self.gcnconv2 = GraphConv(hidden_dim, out_dim)
        
    def forward(self, graph, feat):
        feat = self.gcnconv1(graph, feat)
        feat = F.relu(feat)
        feat = self.gcnconv2(graph, feat)
        return feat
```

Code to extract feature maps for EGUM
```python
# Create list to contain feature maps
node_feat = []
def extract(graph, feat):
    feat = net.gcnconv1(graph, feat)
    feat = F.relu(feat)
    # Please append feat to node_feat list after normalization and activation
    node_feat.append(feat)
    feat = net.gcnconv2(graph, feat)
    node_feat.append(feat)
extract(graph, feat)
```

There's several reminders for you to define a correct function to collect feature maps:
1. Remember to collect feature maps after activation layers and normalization layers, just insert the `node_feat.append()` directly before next message-passing nn layer like `gcnconv2(graph, feat)`
2. Remember that the graph, feat shall be the original graph before the modification(please refers to the overview), which means it shall be the graph and node_feat at the input of initialization functions (Initializer.xxx), but not the return of initialization function, which represent the graph and feat after modification

#### 4.2. load the graph and feat to egum
EGUM provide an useful API to help you load the graph and feature to your customized EGUM model, `load_graph(graph, node_feat, edge_feat)`, please note that the graph shall be the input graph before modification, i.e, the **return graph** from initialzation function (Initializer.xxx). However, the node_feat, and edge_feat, shall be the node_feat and edge_feat you extract before (i.e, it's the excution result of original graph without modification), in Section 4.1:

```python
# if you don't have edge_feat, just ignore this parameter
my_egum.load_graph(graph, node_feat, edge_feat)
```

#### 4.3 run the model
To run the model, as required by pytorch, your input shall meet the requiremnt from `forward` function, and there's some additional requirements from EGUM framework:

Please note that your input graph and features, including node features or edge_features, shall be the subgraph and subgraph features, i.e, the `subgraph, subgraph_node_feat, subgraph_edge_feat` return from initialization funciton, i.e, `Initialization.xxx`

Also, plese kindly note that the output of egum will be in the similar form of initialiation function:
```python
# If no edge_feat is passed in
graph, node_feat = egum(graph, node_feat)
# If edge_feat is passed in
graph, node_feat, edge_feat = egum(graph, node_feat, edge_feat)
```

Please note that output graph, node_feat, edge_feat is of the whole graph, i.e, it's the result of whole graph after GNN inference after modification but not the subgraph.

Congratulation!
## API reference
EGUM provides two class `EGUM` and `initializer` to accelerate your GNN inference. Class `EGUM` is the core of EGUM and it provides all functions for acceleration. Another class, `initializer`, is the class initialize the input for EGUM. Even more, EGUM has a utils class which provides some pytorch tensor logical operations and a tensor based parallel computing hashmap (dictionary).

### EGUM
* `__init__(self, model)`
* `load_graph(self, graph, node_feat, edge_feat=None)`
* `initialize(self, subgraph, node_feat, edge_feat=None)`
* `extend(self, subgraph, node_feat, edge_feat=None)`
* `back(self, subgraph, node_feat, edge_feat=None)`
* `forward(self, subgraph, node_feat, edge_feat=None)`


#### __init__(self, model)
Parameters: **model**: the pytorch model after training

Return: an instance of EGUM

Note that model must be trained
#### load_graph(self, graph, node_feat, edge_feat=None)
Parameters: 
* graph: the raw **input** graph after modification, shall be the subgraph generated by initializer, in form of `dgl.graph`
* node_feat: feature maps before modification, it shall be the startup result, in form of `torch.FloatTensor`
* edge_feat: feature maps of edge before modification, in form of `torch.FloatTensor`

Return: No return

Graph shall be the whole graph after modification, node_feat and edge_feat shall belong to graph without modification, see Tutorial Section 4.1 and 4.2
#### initialize(self, subgraph, node_feat, edge_feat=None)
Parameters:
* subgraph: the subgraph from initialization function in form of `dgl.graph` 
* node_feat: node_feat from initialization function in form of `torch.FloatTensor`
* edge_feat: edge_feat from initialization function in form of `torch.FloatTensor`, not compulsory, can be ignored if no edge_feat from dataset

Return: No return

Note that node_feat and edge_feat must be features belongs to subgraph
#### extend(self, subgraph, node_feat, edge_feat=None)
Parameters:
* subgraph: the subgraph from initialization function in form of `dgl.graph` 
* node_feat: node_feat from initialization function in form of `torch.FloatTensor`
* edge_feat: edge_feat from initialization function in form of `torch.FloatTensor`, not compulsory, can be ignored if no edge_feat from dataset

Return:
* subgraph: subgraph after extend by a step
* node_feat: node_feat of subgraph after extend by a step
* edge_feat: edge_feat of subgraph after extend by a step, will be returned only when edge_feat is input

This function will extend the graph by a step.
#### back(self, subgraph, node_feat, edge_feat=None)
Parameters:
* subgraph: the subgraph from initialization function in form of `dgl.graph` 
* node_feat: node_feat from initialization function in form of `torch.FloatTensor`
* edge_feat: edge_feat from initialization function in form of `torch.FloatTensor`, not compulsory, can be ignored if no edge_feat from dataset

Return:
* graph: whole graph after execution
* node_feat: node_feat of the whole graph after GNN process
* edge_feat: edge_feat of the whole graph after GNN process, will be returned only when edge_feat is input

Please note that input is the subgraph, with its node and edge features, but the return is the whole graph after modification, with the node features and edge features of the whole graph after modification.

#### forward(self, kwargs)
This is an empty function left for user to implement, please refers to Tutorial Section 3 to learn how to implement EGUM.
