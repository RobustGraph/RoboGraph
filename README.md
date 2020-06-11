# RoboGraph

## Usage

### Installation

The project requires python with version 3.7+, and use pip to install required packages
* install pytorch from [link](https://pytorch.org/get-started/locally/)
* install pytorch_geometric from [link](https://github.com/rusty1s/pytorch_geometric#installation)
* install cplex and docplex

For example, in the cpu only machine:

```shell
conda install python=3.7
conda install pytorch torchvision cpuonly -c pytorch
pip install torch-scatter==latest+cpu torch-sparse==latest+cpu torch-cluster==latest+cpu torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric
pip install qpsolvers, sympy, nsopy
```

After install cplex, install docplex:

```shell
conda install -c ibmdecisionoptimization docplex
```

## Datasets

TU of Dortmund has a collection of benchmark data sets for graph kernels.
  
* multi-graph data set
* node features (applied to some data sets)
* link features (applied to some data sets)

Reference: [Benchmark Data Sets for Graph Kernel](http://graphkernels.cs.tu-dortmund.de/)

### Selected Datasets

* setting: 30% for training, 20% for validation and 50% for testing

| NAME     | No. of Graph | No. of Classes | Avg. No. of Nodes | Avg. No. of Edges | No. of node features |
| -------- | ------------ | -------------- | ----------------- | ----------------- | -------------------- |
| ENZYMES  | 600          | 6              | 32.63             | 62.14             | 21                   |
| PROTEINS | 1113         | 2              | 39.06             | 72.82             | 4                    |
| NCI1     | 4110         | 2              | 29.87             | 32.30             | -                    |
| MUTAG    | 188          | 2              | 17.93             | 19.79             | -                    |

<!-- | DD       | 1178         | 2              | 284.32            | 715.66            | 89                   | -->

| dataset  | # of graphs | # of label | # of features | min edge | max edge | median edge | min node | max node | median node |
| -------- | ----------- | ---------- | ------------- | -------- | -------- | ----------- | -------- | -------- | ----------- |
| ENZYMES  | 600         | 6          | 21            | 2        | 298      | 120         | 2        | 126      | 32          |
| NCI1     | 4110        | 2          | 37            | 4        | 238      | 58          | 3        | 111      | 27          |
| PROTEINS | 1113        | 2          | 4             | 10       | 2098     | 98          | 4        | 620      | 26          |
| MUTAG    | 188         | 2          | 7             | 20       | 66       | 38          | 10       | 28       | 17          |

<!-- | DD       | 1178        | 2          | 89            | 126      | 28534    | 1221        | 30       | 5748     | 241         | -->