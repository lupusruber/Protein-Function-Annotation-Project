import pandas as pd
import os
import random
import torch

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

import argparse

# PROTEIN_DATASET_ROOT = "/home/lupusruber/root/projects/ppi/PPI/protein_dataset"
# PROCESSED_DATA_PATH = '/home/lupusruber/root/projects/ppi/PPI/data_preparators'

# home/lupusruber/root/projects/ppi/PPI/protein_dataset/ppi/human_ppi_700.txt
# /home/lupusruber/root/projects/ppi/PPI/protein_dataset/ppi/human_ppi_700.txt


def clear_filtered_edges(t: int, ont: str, train_or_test: str = 'train') -> None:
    fp_positive = f"{PROTEIN_DATASET_ROOT}/benchmark/human_ppi_{t}/{train_or_test}_{ont}_filtered_positive_edges.txt"
    fp_negative = f"{PROTEIN_DATASET_ROOT}/benchmark/human_ppi_{t}/{train_or_test}_{ont}_filtered_negative_edges.txt"

    if os.path.exists(fp_positive):
        os.remove(fp_positive)
    if os.path.exists(fp_negative):
        os.remove(fp_negative)


def get_ppi_edges(t: int) -> pd.DataFrame:
    
    ppi_edges = pd.read_table(f"{PROTEIN_DATASET_ROOT}/ppi/human_ppi_{t}.txt")
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
    return ppi_edges


def get_annotation_edges(
    t: int, ont: str, ppi_edges: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if ont == "all":
        gt_edges_bp = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/goa/t0/human_ppi_{t}_BP_anno_filtered.txt"
        )
        gt_edges_mf = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/goa/t0/human_ppi_{t}_MF_anno_filtered.txt"
        )
        gt_edges_cc = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/goa/t0/human_ppi_{t}_CC_anno_filtered.txt"
        )
        gt_edges = pd.concat((gt_edges_bp, gt_edges_mf, gt_edges_cc))
        gt_edges_bp_t2 = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/goa/t1/human_ppi_{t}_BP_anno_filtered.txt"
        )
        gt_edges_mf_t2 = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/goa/t1/human_ppi_{t}_MF_anno_filtered.txt"
        )
        gt_edges_cc_t2 = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/goa/t1/human_ppi_{t}_CC_anno_filtered.txt"
        )
        gt_edges_t2 = pd.concat((gt_edges_bp_t2, gt_edges_mf_t2, gt_edges_cc_t2))
    else:
        gt_edges = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/goa/t0/human_ppi_{t}_{ont}_anno_filtered.txt"
        )
        gt_edges_t2 = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/goa/t1/human_ppi_{t}_{ont}_anno_filtered.txt"
        )

    gt_edges.index = [i for i in range(len(ppi_edges), len(ppi_edges) + len(gt_edges))]
    gt_edges.columns = ["source", "target"]
    gt_edges_t2.columns = ["source", "target"]

    return gt_edges, gt_edges_t2


def get_protein_embeddings(ppi_edges: pd.DataFrame) -> pd.DataFrame:

    proteins_df = pd.DataFrame(
        set(ppi_edges["source"].values.tolist() + ppi_edges["target"].values.tolist()),
        columns=["Protein"],
    )

    protein_nodes_df = pd.read_csv(
        f"{PROTEIN_DATASET_ROOT}/features/protein_embeddings.csv"
    ).merge(proteins_df)
    protein_nodes_df.index = protein_nodes_df["Protein"].values
    protein_nodes_df.drop(["Protein"], axis=1, inplace=True)

    return protein_nodes_df


def get_gene_onotology_term_embeddings(
    gt_edges: pd.DataFrame, gt_edges_t2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    t1_terms_list = set(list(gt_edges["target"].values))
    t2_terms_list = set(list(gt_edges_t2["target"].values))
    terms_df = pd.DataFrame(
        set(list(t1_terms_list) + list(t2_terms_list)), columns=["go_id"]
    )
    go_term_nodes_df = pd.read_csv(
        f"{PROTEIN_DATASET_ROOT}/features/terms_embeddings.csv"
    ).merge(terms_df)
    go_term_nodes_df.index = go_term_nodes_df["go_id"].values
    go_term_nodes_df.drop(["go_id"], axis=1, inplace=True)

    return terms_df, go_term_nodes_df


def get_positive_and_negative_edges(
    t: int,
    ont: str,
    terms_df: pd.DataFrame,
    gt_edges: pd.DataFrame,
    gt_edges_t2: pd.DataFrame,
    train_or_test: str = 'train'
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if ont == "all":
        train_proteins_bp = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/benchmark/human_ppi_{t}/{train_or_test}_BP_filtered.txt"
        )["protein_id"].values.tolist()
        train_proteins_mf = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/benchmark/human_ppi_{t}/{train_or_test}_MF_filtered.txt"
        )["protein_id"].values.tolist()
        train_proteins_cc = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/benchmark/human_ppi_{t}/{train_or_test}_CC_filtered.txt"
        )["protein_id"].values.tolist()
        train_proteins = train_proteins_bp + train_proteins_mf + train_proteins_cc

    else:
        train_proteins = pd.read_table(
            f"{PROTEIN_DATASET_ROOT}/benchmark/human_ppi_{t}/{train_or_test}_{ont}_filtered.txt"
        )["protein_id"].values.tolist()

    all_terms = list(terms_df["go_id"].values.tolist())
    positive_edges, negative_edges = [], []

    for protein in train_proteins:
        t1_anno = gt_edges.loc[gt_edges["source"] == protein]["target"].values.tolist()
        t2_anno = gt_edges_t2.loc[gt_edges_t2["source"] == protein][
            "target"
        ].values.tolist()
        # t1_anno, t2_anno = [gt_edges.query(f'source == "{protein}"')['target'].tolist(), gt_edges_t2.query(f'source == "{protein}"')['target'].tolist()]
        protein_pe = [[protein, anno] for anno in set(t1_anno).difference(set(t2_anno))]
        protein_ne = [
            [protein, anno]
            for anno in set(all_terms).difference(set(t1_anno + t2_anno))
        ]
        positive_edges.extend(protein_pe)
        negative_edges.extend(random.sample(protein_ne, len(protein_pe)))

    # positive_edges_df = pd.DataFrame(positive_edges, columns=["protein", "go_id"])
    # negative_edges_df = pd.DataFrame(negative_edges, columns=["protein", "go_id"])

    positive_edges_df = pd.DataFrame(positive_edges, columns=["source", "target"])
    negative_edges_df = pd.DataFrame(negative_edges, columns=["source", "target"])

    return positive_edges_df, negative_edges_df


def remove_edges_from_graph(
    positive_edges_df: pd.DataFrame,
    negative_edges_df: pd.DataFrame,
    gt_edges: pd.DataFrame,
    ppi_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    edges_to_remove = pd.concat((positive_edges_df, negative_edges_df))

    gt_edges_removed = (
        gt_edges.merge(edges_to_remove, indicator="i", how="outer")
        .query('i == "left_only"')
        .drop(labels=["i"], axis=1)
    )
    gt_edges_removed.index = [
        i for i in range(len(ppi_edges), len(ppi_edges) + len(gt_edges_removed))
    ]

    return gt_edges_removed, edges_to_remove


def create_index_mapping(
    protein_nodes_df: pd.DataFrame, go_term_nodes_df: pd.DataFrame
) -> tuple[dict, dict]:

    protein_map = {protein_nodes_df.index[i]: i for i in range(len(protein_nodes_df))}
    go_term_map = {go_term_nodes_df.index[i]: i for i in range(len(go_term_nodes_df))}

    return protein_map, go_term_map


def create_hetero_data_object(
    protein_nodes_df: pd.DataFrame,
    go_term_nodes_df: pd.DataFrame,
    ppi_edges: pd.DataFrame,
    gt_edges_removed: pd.DataFrame,
) -> HeteroData:

    graph_data = HeteroData()

    # Add graph nodes with their features
    graph_data["protein"].x = torch.tensor(protein_nodes_df.values)
    graph_data["go_term"].x = torch.tensor(go_term_nodes_df.values)

    protein_map, go_term_map = create_index_mapping(
        protein_nodes_df=protein_nodes_df, go_term_nodes_df=go_term_nodes_df
    )

    ppi_edge_index = [
        [protein_map[p] for p in ppi_edges["source"]],
        [protein_map[p] for p in ppi_edges["target"]],
    ]
    annotation_edge_index = [
        [protein_map[p] for p in gt_edges_removed["source"]],
        [go_term_map[gt] for gt in gt_edges_removed["target"]],
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

    return graph_data


def create_train_loader(
    positive_edges_df: pd.DataFrame,
    negative_edges_df: pd.DataFrame,
    protein_nodes_df: pd.DataFrame,
    go_term_nodes_df: pd.DataFrame,
    edges_to_remove: pd.DataFrame,
    batch_size: int,
    graph_data: HeteroData,
) -> LinkNeighborLoader:

    labels_train = pd.DataFrame(
        [1] * len(positive_edges_df) + [0] * len(negative_edges_df), columns=["label"]
    )

    protein_map, go_term_map = create_index_mapping(
        protein_nodes_df=protein_nodes_df, go_term_nodes_df=go_term_nodes_df
    )

    edges_to_remove_index = [
        [protein_map[p] for p in edges_to_remove["source"]],
        [go_term_map[gt] for gt in edges_to_remove["target"]],
    ]

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

    return train_loader


def train_graph_data_preparation(
    t: int, ont: str, batch_size: int, train_or_test: str = 'train'
) -> tuple[LinkNeighborLoader, HeteroData]:

    clear_filtered_edges(t=t, ont=ont, train_or_test=train_or_test)

    # Load PPI edges
    ppi_edges = get_ppi_edges(t=t)

    # Load annotation edges
    gt_edges, gt_edges_t2 = get_annotation_edges(t=t, ont=ont, ppi_edges=ppi_edges)

    # Load protein embeddings
    protein_nodes_df = get_protein_embeddings(ppi_edges=ppi_edges)

    # Load term embeddings
    terms_df, go_term_nodes_df = get_gene_onotology_term_embeddings(
        gt_edges=gt_edges, gt_edges_t2=gt_edges_t2
    )

    # Create positive and negative edges
    positive_edges_df, negative_edges_df = get_positive_and_negative_edges(
        t=t, ont=ont, terms_df=terms_df, gt_edges=gt_edges, gt_edges_t2=gt_edges_t2, train_or_test=train_or_test
    )

    # Remove edges
    gt_edges_removed, edges_to_remove = remove_edges_from_graph(
        positive_edges_df=positive_edges_df,
        negative_edges_df=negative_edges_df,
        gt_edges=gt_edges,
        ppi_edges=ppi_edges,
    )

    # Create graph
    graph_data = create_hetero_data_object(
        protein_nodes_df=protein_nodes_df,
        go_term_nodes_df=go_term_nodes_df,
        ppi_edges=ppi_edges,
        gt_edges_removed=gt_edges_removed,
    )

    print(graph_data)

    # train_loader = create_train_loader(
    #     positive_edges_df=positive_edges_df,
    #     negative_edges_df=negative_edges_df,
    #     protein_nodes_df=protein_nodes_df,
    #     go_term_nodes_df=go_term_nodes_df,
    #     edges_to_remove=edges_to_remove,
    #     batch_size=batch_size,
    #     graph_data=graph_data,
    # )

    
    # torch.save(train_loader, f'{DATA_PREPARATOR_PATH}/train_loader.pt')


    # return train_loader, graph_data
    return None, graph_data


def get_whole_dataset(t: int, ont: str) -> None:
    _, train_graph_data = train_graph_data_preparation(t=t, ont=ont, batch_size=32, train_or_test='train')
    _, test_graph_data = train_graph_data_preparation(t=t, ont=ont, batch_size=32, train_or_test='test')

    # torch.save(train_graph_data, f'{DATA_PREPARATOR_PATH}/train_graph_data_{ont}_{t}.pt')
    # torch.save(test_graph_data, f'{DATA_PREPARATOR_PATH}/test_graph_data_{ont}_{t}.pt')


    whole_data = HeteroData()

    whole_data['protein'] = train_graph_data['protein']
    whole_data['protein']['num_nodes'] = whole_data['protein']['x'].shape[0]
    whole_data['go_term'] = train_graph_data['go_term']
    whole_data['protein', 'interacts', 'protein'] = train_graph_data['protein', 'interacts', 'protein']

    train_annotations = train_graph_data['protein', 'annotated', 'go_term']
    test_annotations = test_graph_data['protein', 'annotated', 'go_term']

    whole_data['protein', 'annotated', 'go_term'] = train_annotations.concat(test_annotations)

    print(whole_data)

    torch.save(whole_data, f'{PROCESSED_DATA_PATH}/whole_graph_data_{ont}_{t}.pt')



def generate_all_datasets():
    ontologies = ['BP', 'CC', 'MF']
    t_values = [700, 900]
    for ont in ontologies:
        for t in t_values:
            get_whole_dataset(t=t, ont=ont)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--raw-data", type=str, help="full path of the protein dataset")
    argparser.add_argument("--generated-dataset", type=str,  help="full path location of saved heterodata objects")
    
    args = argparser.parse_args()
    
    PROTEIN_DATASET_ROOT = args.raw_data
    PROCESSED_DATA_PATH = args.generated_dataset
    
    generate_all_datasets()
