from typing import Any
import pandas as pd
import os
from dotenv import load_dotenv
import random
import torch

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader


def train_graph_data_preparation(
    t: int, ont: str, batch_size: int
) -> tuple[LinkNeighborLoader, HeteroData, tuple[Any]]:

    load_dotenv()
    root = os.getenv("ROOT", "")

    fp_positive = f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_{ont}_filtered_positive_edges.txt"
    fp_negative = f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_{ont}_filtered_negative_edges.txt"

    if os.path.exists(fp_positive):
        os.remove(fp_positive)
    if os.path.exists(fp_negative):
        os.remove(fp_negative)

    # Load PPI edges
    # /home/lupusruber/root/projects/pold_code/protein_dataset/ppi/human_ppi_900.txt
    ppi_edges = pd.read_table(f"{root}old_code/protein_dataset/ppi/human_ppi_{t}.txt")
    ppi_edges.columns = [
        "source",
        "target",
        "neighborhood",
        "fusion",
        "cooccurence",
        "coexpression",
        "experimental",
        "database",
        "textmining",
        "combined_score",
    ]

    # Load annotation edges
    if ont == "all":
        gt_edges_bp = pd.read_table(
            f"{root}old_code/protein_dataset/goa/t0/human_ppi_{t}_BP_anno_filtered.txt"
        )
        gt_edges_mf = pd.read_table(
            f"{root}old_code/protein_dataset/goa/t0/human_ppi_{t}_MF_anno_filtered.txt"
        )
        gt_edges_cc = pd.read_table(
            f"{root}old_code/protein_dataset/goa/t0/human_ppi_{t}_CC_anno_filtered.txt"
        )
        gt_edges = pd.concat((gt_edges_bp, gt_edges_mf, gt_edges_cc))
        gt_edges_bp_t2 = pd.read_table(
            f"{root}old_code/protein_dataset/goa/t1/human_ppi_{t}_BP_anno_filtered.txt"
        )
        gt_edges_mf_t2 = pd.read_table(
            f"{root}old_code/protein_dataset/goa/t1/human_ppi_{t}_MF_anno_filtered.txt"
        )
        gt_edges_cc_t2 = pd.read_table(
            f"{root}old_code/protein_dataset/goa/t1/human_ppi_{t}_CC_anno_filtered.txt"
        )
        gt_edges_t2 = pd.concat((gt_edges_bp_t2, gt_edges_mf_t2, gt_edges_cc_t2))
    else:
        gt_edges = pd.read_table(
            f"{root}old_code/protein_dataset/goa/t0/human_ppi_{t}_{ont}_anno_filtered.txt"
        )
        gt_edges_t2 = pd.read_table(
            f"{root}old_code/protein_dataset/goa/t1/human_ppi_{t}_{ont}_anno_filtered.txt"
        )
    gt_edges.index = [i for i in range(len(ppi_edges), len(ppi_edges) + len(gt_edges))]
    gt_edges.columns = ["source", "target"]
    gt_edges_t2.columns = ["source", "target"]

    # Load protein embeddings
    proteins_df = pd.DataFrame(
        set(ppi_edges["source"].values.tolist() + ppi_edges["target"].values.tolist()),
        columns=["Protein"],
    )
    protein_nodes_df = pd.read_csv(
        f"{root}old_code/protein_dataset/features/protein_embeddings.csv"
    ).merge(proteins_df)
    protein_nodes_df.index = protein_nodes_df["Protein"].values
    protein_nodes_df.drop(["Protein"], axis=1, inplace=True)

    # Load term embeddings
    t1_terms_list = set(list(gt_edges["target"].values))
    t2_terms_list = set(list(gt_edges_t2["target"].values))
    terms_df = pd.DataFrame(
        set(list(t1_terms_list) + list(t2_terms_list)), columns=["go_id"]
    )
    go_term_nodes_df = pd.read_csv(
        f"{root}old_code/protein_dataset/features/terms_embeddings.csv"
    ).merge(terms_df)
    go_term_nodes_df.index = go_term_nodes_df["go_id"].values
    go_term_nodes_df.drop(["go_id"], axis=1, inplace=True)

    # Create positive and negative edges
    if os.path.exists(
        f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_{ont}_filtered_positive_edges.txt"
    ):
        positive_edges_df = pd.read_csv(
            f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_{ont}_filtered_positive_edges.txt",
            sep="\t",
        )
        positive_edges_df.columns = ["source", "target"]
        negative_edges_df = pd.read_csv(
            f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_{ont}_filtered_negative_edges.txt",
            sep="\t",
        )
        negative_edges_df.columns = ["source", "target"]
    else:
        if ont == "all":
            train_proteins_bp = pd.read_table(
                f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_BP_filtered.txt"
            )["protein_id"].values.tolist()
            train_proteins_mf = pd.read_table(
                f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_MF_filtered.txt"
            )["protein_id"].values.tolist()
            train_proteins_cc = pd.read_table(
                f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_CC_filtered.txt"
            )["protein_id"].values.tolist()
            train_proteins = train_proteins_bp + train_proteins_mf + train_proteins_cc
        else:
            train_proteins = pd.read_table(
                f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_{ont}_filtered.txt"
            )["protein_id"].values.tolist()
        all_terms = list(terms_df["go_id"].values.tolist())
        positive_edges, negative_edges = [], []
        for protein in train_proteins:
            t1_anno = gt_edges.loc[gt_edges["source"] == protein][
                "target"
            ].values.tolist()
            t2_anno = gt_edges_t2.loc[gt_edges_t2["source"] == protein][
                "target"
            ].values.tolist()
            # t1_anno, t2_anno = [gt_edges.query(f'source == "{protein}"')['target'].tolist(), gt_edges_t2.query(f'source == "{protein}"')['target'].tolist()]
            protein_pe = [
                [protein, anno] for anno in set(t1_anno).difference(set(t2_anno))
            ]
            protein_ne = [
                [protein, anno]
                for anno in set(all_terms).difference(set(t1_anno + t2_anno))
            ]
            positive_edges.extend(protein_pe)
            negative_edges.extend(random.sample(protein_ne, len(protein_pe)))
        positive_edges_df = pd.DataFrame(positive_edges, columns=["protein", "go_id"])

        positive_edges_df.to_csv(
            f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_{ont}_filtered_positive_edges.txt",
            sep="\t",
            index=False,
        )
        negative_edges_df = pd.DataFrame(negative_edges, columns=["protein", "go_id"])

        negative_edges_df.to_csv(
            f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/train_{ont}_filtered_negative_edges.txt",
            sep="\t",
            index=False,
        )

    positive_edges_df = pd.DataFrame(positive_edges, columns=["source", "target"])
    negative_edges_df = pd.DataFrame(negative_edges, columns=["source", "target"])
    # Remove edges
    labels_train = pd.DataFrame(
        [1] * len(positive_edges_df) + [0] * len(negative_edges_df), columns=["label"]
    )
    edges_to_remove = pd.concat((positive_edges_df, negative_edges_df))
    gt_edges_removed = (
        gt_edges.merge(edges_to_remove, indicator="i", how="outer")
        .query('i == "left_only"')
        .drop(labels=["i"], axis=1)
    )
    gt_edges_removed.index = [
        i for i in range(len(ppi_edges), len(ppi_edges) + len(gt_edges_removed))
    ]

    protein_map = {protein_nodes_df.index[i]: i for i in range(len(protein_nodes_df))}
    go_term_map = {go_term_nodes_df.index[i]: i for i in range(len(go_term_nodes_df))}

    graph_data = HeteroData()

    # Add graph nodes with their features
    graph_data["protein"].x = torch.tensor(protein_nodes_df.values)
    graph_data["go_term"].x = torch.tensor(go_term_nodes_df.values)

    ppi_edge_index = [
        [protein_map[p] for p in ppi_edges["source"]],
        [protein_map[p] for p in ppi_edges["target"]],
    ]
    annotation_edge_index = [
        [protein_map[p] for p in gt_edges_removed["source"]],
        [go_term_map[gt] for gt in gt_edges_removed["target"]],
    ]
    edges_to_remove_index = [
        [protein_map[p] for p in edges_to_remove["source"]],
        [go_term_map[gt] for gt in edges_to_remove["target"]],
    ]

    assert len(ppi_edge_index[0]) == len(ppi_edge_index[1])
    assert len(annotation_edge_index[0]) == len(annotation_edge_index[1])

    # Add graph edges
    graph_data["protein", "interacts", "protein"].edge_index = torch.tensor(
        ppi_edge_index
    )
    graph_data["protein", "annotated", "go_term"].edge_index = torch.tensor(
        annotation_edge_index
    )

    # Add graph attributes
    ppi_attributes = torch.tensor(ppi_edges.drop(["source", "target"], axis=1).values)
    graph_data["protein", "interacts", "protein"].edge_attr = ppi_attributes

    print(graph_data)
    print(labels_train.values)

    train_loader = LinkNeighborLoader(
        data=graph_data,
        num_neighbors=[-1],
        edge_label_index=(
            ("protein", "annotated", "go_term"),
            torch.tensor(edges_to_remove_index),
        ),
        edge_label=torch.tensor(labels_train.values),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, graph_data, graph_data.metadata()
