from typing import Any, Union
import torch
from torch.nn import ReLU, LogSoftmax
from torch import Tensor
from torch_geometric.nn import HANConv, LayerNorm, Linear, SAGEConv, to_hetero
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from tqdm import tqdm


class GraphSage(torch.nn.Module):
    def __init__(self, hidden_channels: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(
        self,
        metadata: tuple[Any],
        dim_in: Union[int, None],
        dim_h: int,
        dim_out: int,
        heads: int,
        learning_rate: float,
        dropout: float,
    ) -> None:
        super().__init__()
        torch.manual_seed(1234567)

        self.gat1 = HANConv(
            dim_in, dim_h, heads=heads, dropout=dropout, metadata=metadata
        )
        self.gat2 = HANConv(
            dim_h, dim_h, heads=heads, dropout=dropout, metadata=metadata
        )
        self.gat3 = HANConv(
            dim_h, dim_h, heads=heads, dropout=dropout, metadata=metadata
        )

        self.norm = LayerNorm(dim_h)
        self.linear = Linear(dim_h, dim_h)

        self.relu = ReLU()
        self.softmax = LogSoftmax(dim=1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        h = self.gat1(x, edge_index)
        h["protein"] = self.relu(h["protein"])
        h["go_term"] = self.relu(h["go_term"])
        # h['protein'] = self.norm(h['protein'])
        # h['go_term'] = self.norm(h['go_term'])

        h = self.gat2(h, edge_index)
        h["protein"] = self.relu(h["protein"])
        h["go_term"] = self.relu(h["go_term"])
        # h['protein'] = self.norm(h['protein'])
        # h['go_term'] = self.norm(h['go_term'])

        h = self.gat3(h, edge_index)
        h["protein"] = self.linear(h["protein"])
        h["go_term"] = self.linear(h["go_term"])
        # h['protein'] = self.linear(h['protein'])
        # h['go_term'] = self.linear(h['go_term'])

        return h


class Classifier(torch.nn.Module):
    def forward(
        self, protein: Tensor, go_term: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        edge_feat_protein = protein[edge_label_index[0]]
        edge_feat_go_term = go_term[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_protein * edge_feat_go_term).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        metadata: tuple[Any],
        dim_h: int,
        dim_in: Union[int, None] = None,
        data: HeteroData = None,
        dim_out: int = 1,
        heads: int = 4,
        learning_rate: float = 0.001,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if model_name == "GAT":
            self.gat = GAT(
                metadata=metadata,
                dim_in=dim_in,
                dim_h=dim_h,
                dim_out=dim_out,
                heads=heads,
                learning_rate=learning_rate,
                dropout=dropout,
            )
            self.classifier = Classifier()
            self.linear = Linear(dim_h, dim_out)
            self.model_name = "GAT"
        else:
            self.protein_lin = torch.nn.Linear(1000, 64)
            self.protein_emb = torch.nn.Embedding(data["protein"].num_nodes, dim_h)
            self.go_term_emb = torch.nn.Embedding(data["go_term"].num_nodes, dim_h)
            # Instantiate homogeneous GNN:
            self.het_graph_sage = GraphSage(hidden_channels=dim_h, dropout_rate=dropout)
            # Convert GNN model into a heterogeneous variant:
            self.het_graph_sage = to_hetero(
                self.het_graph_sage, metadata=data.metadata()
            )
            self.classifier = Classifier()
            self.model_name = "GraphSage"

    def forward(self, graph_data: HeteroData) -> Tensor:
        if self.model_name == "GAT":
            x_dict = {
                "protein": graph_data["protein"].x.to(torch.float32),
                "go_term": graph_data["go_term"].x.to(torch.float32),
            }
        else:
            x_dict = {
                "protein": self.protein_emb(graph_data["protein"].n_id),
                "go_term": self.protein_lin(graph_data["go_term"].x)
                + self.go_term_emb(graph_data["go_term"].n_id),
            }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        if self.model_name == "GAT":
            x_dict = self.gat(x_dict, graph_data.edge_index_dict)
        else:
            x_dict = self.het_graph_sage(x_dict, graph_data.edge_index_dict)

        pred = self.classifier(
            x_dict["protein"],
            x_dict["go_term"],
            graph_data["protein", "annotated", "go_term"].edge_label_index,
        )

        if self.model_name == "GAT":
            # pred = self.linear(pred)
            pred = torch.relu(pred)
            pred = F.softmax(pred, dim=-1)

        return pred


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        metadata: tuple[Any],
        dim_in: int,
        dim_h: int,
        dim_out: int,
        learning_rate: float,
        dropout: float,
    ) -> None:
        super().__init__()
        torch.manual_seed(1234567)

        # Define the convolutional layers
        self.conv1 = SAGEConv((-1, -1), dim_h, aggr="mean")
        self.conv2 = SAGEConv((-1, -1), dim_h, aggr="mean")
        self.conv3 = SAGEConv((-1, -1), dim_h, aggr="mean")

        self.norm = LayerNorm(dim_h)
        self.linear = Linear(dim_h, dim_out)

        self.relu = ReLU()
        self.softmax = LogSoftmax(dim=1)

        # Convert the model to a heterogeneous one
        self.conv1 = to_hetero(self.conv1, metadata)
        self.conv2 = to_hetero(self.conv2, metadata)
        self.conv3 = to_hetero(self.conv3, metadata)

    def forward(self, x_dict: HeteroData, edge_index_dict: HeteroData) -> Tensor:
        h = self.conv1(x_dict, edge_index_dict)
        h = {key: self.relu(h[key]) for key in h}

        h = self.conv2(h, edge_index_dict)
        h = {key: self.relu(h[key]) for key in h}

        h = self.conv3(h, edge_index_dict)
        h = {key: self.linear(h[key]) for key in h}

        return h
