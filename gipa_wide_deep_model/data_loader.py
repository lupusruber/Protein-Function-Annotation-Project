import torch
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl
from collections import defaultdict
from torch.utils.data import random_split
from torch_geometric.utils import to_dgl
from collections import Counter
import numpy as np

interval_0_1 = [0.001, 1]
interval_3 = [0.001, 0.7, 1]
interval_4 = [0.001, 0.1, 0.2, 1]
interval_12 = [0.001, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1]
interval_15 = [0.001, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1]
simple_inter = [interval_0_1, interval_4, interval_0_1, interval_4, interval_12, interval_12, interval_3, interval_15]


def change_pyg_hetero_data(path: str):
  data = torch.load(path)
  proteins, go_terms = data['protein', 'annotated', 'go_term']['edge_index']

  counter = Counter(go_terms.numpy())
  n_most_common = 100

  set_of_most_common = set(np.array(counter.most_common(n_most_common))[:, 0])
  mapper = dict(zip(set_of_most_common, range(n_most_common)))

  adj_matrix = torch.zeros((proteins.max().item()+1,n_most_common), dtype=torch.int64)

  for x, y in data['protein', 'annotated', 'go_term']['edge_index'].transpose(0, 1):
      y_value = y.item() 
      if y_value in set_of_most_common:
        adj_matrix[x, mapper[y_value]] = 1

  del data['protein', 'annotated', 'go_term']
  del data['go_term']


  data['protein']['labels'] = adj_matrix

  return data, adj_matrix


def train_val_test_split(n_nodes: int) -> tuple[torch.tensor]:
  train_idx, val_idx, test_idx = random_split(range(n_nodes), [0.8, 0.1, 0.1])
  train_idx = torch.tensor(train_idx)
  val_idx = torch.tensor(val_idx)
  test_idx = torch.tensor(test_idx)

  return train_idx, val_idx, test_idx



def trans_edge_fea_to_sparse(raw_edge_fea, graph, interval: list, is_log=False):
    edge_fea_list = []
    for i in range(8):
        print("Process edge feature == %d " % i)
        res = torch.reshape((raw_edge_fea[:, i] == 0.001).float(), [-1, 1])
        edge_fea_list.append(res)
        for j in range(1, len(interval[i])):
            small, big = float(interval[i][j - 1]), float(interval[i][j])
            print("process interval %0.3f < x <= %0.3f " % (small, big))
            cond = torch.logical_and((raw_edge_fea[:, i] > small), (raw_edge_fea[:, i] <= big))
            edge_fea_list.append(torch.reshape(cond.float(), [-1, 1]))
    sparse = torch.concat(edge_fea_list, dim=-1)
    print(sparse.size())
    graph.edata.update({"sparse": sparse})
    graph.update_all(fn.copy_e("sparse", "sparse_c"), fn.sum("sparse_c", "sparse_f" if is_log else "sparse"))
    if is_log:
        graph.apply_nodes(lambda nodes: {"sparse": torch.log2(nodes.data['sparse_f'] + 1)})
        del graph.ndata["sparse_f"]
    return sparse


def compute_norm(graph):
    degs = graph.in_degrees().float().clamp(min=1)
    deg_isqrt = torch.pow(degs, -0.5)

    degs = graph.in_degrees().float().clamp(min=1)
    deg_sqrt = torch.pow(degs, 0.5)

    return deg_sqrt, deg_isqrt


def add_changed_data(graph, labels, train_idx, test_idx, val_idx, evaluator):
  
  # Sample 10% of nodes
  num_nodes = graph.number_of_nodes()
  num_sampled_nodes = int(num_nodes * 0.1)
  sampled_nodes = torch.randperm(num_nodes)[:num_sampled_nodes]

  # mapper
  mapper = {
    node: index for index, node in enumerate(sampled_nodes.tolist())
  }

  # Create the sampled subgraph
  sampled_graph = graph.subgraph(sampled_nodes)
  sampled_labels = labels[sampled_nodes]

  # Filter train, val, test indices for sampled nodes
  def filter_indices(indices):
      return torch.Tensor([mapper[idx.item()] for idx in indices if idx in sampled_nodes]).to(dtype=torch.int64)

  sampled_train_idx = filter_indices(train_idx)
  sampled_val_idx = filter_indices(val_idx)
  sampled_test_idx = filter_indices(test_idx)

  # Print statistics for the sampled graph
  print(f"Nodes : {sampled_graph.number_of_nodes()}\n"
        f"Edges: {sampled_graph.number_of_edges()}\n"
        f"Train nodes: {len(sampled_train_idx)}\n"
        f"Val nodes: {len(sampled_val_idx)}\n"
        f"Test nodes: {len(sampled_test_idx)}")
    
  return sampled_graph, sampled_labels, sampled_train_idx, sampled_val_idx, sampled_test_idx, evaluator
 


def load_data(dataset, root_path):
    #data = DglNodePropPredDataset(name=dataset, root=root_path)
    evaluator = None
    pyg_data, labels = change_pyg_hetero_data('/content/graph_data.pt')
    pyg_data = pyg_data.to_homogeneous()

    del pyg_data['node_type']
    del pyg_data['edge_type']
    graph = to_dgl(pyg_data)

    graph.edata['feat'] = graph.edata['edge_attr']
    graph.edata['feat'] = graph.edata['feat'].to(dtype=torch.float32)

    del graph.edata['edge_attr']
    #n_protein_nodes = graph.get_node_storage(ntype='protein', key='feat').storage.shape[0]
    n_protein_nodes = graph.num_nodes()
    train_idx, val_idx, test_idx = train_val_test_split(n_protein_nodes)
    
    #evaluator = Evaluator(name=dataset)
    #train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    

    print(f"Nodes : {graph.number_of_nodes()}\n"
          f"Edges: {graph.number_of_edges()}\n"
          f"Train nodes: {len(train_idx)}\n"
          f"Val nodes: {len(val_idx)}\n"
          f"Test nodes: {len(test_idx)}")
    
    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph, labels, edge_agg_as_feat=True, user_adj=False, user_avg=False, sparse_encoder: str = None):
    if edge_agg_as_feat:
        graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))

    if sparse_encoder is not None and len(sparse_encoder) > 0:
        is_log = sparse_encoder.find("log") > -1
        edge_sparse = trans_edge_fea_to_sparse(graph.edata['feat'], graph, simple_inter, is_log)

        if sparse_encoder.find("edge_sparse") > -1:
            graph.edata.update({"feat": edge_sparse})
        del graph.edata["sparse"]

    if user_adj or user_avg:
        deg_sqrt, deg_isqrt = compute_norm(graph)
        if user_adj:
            graph.srcdata.update({"src_norm": deg_isqrt})
            graph.dstdata.update({"dst_norm": deg_sqrt})
            graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))

        if user_avg:
            graph.srcdata.update({"src_norm": deg_isqrt})
            graph.dstdata.update({"dst_norm": deg_isqrt})
            graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))

    graph.create_formats_()
    print(graph.ndata.keys())
    print(graph.edata.keys())
    return graph, labels
