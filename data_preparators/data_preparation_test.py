import pandas as pd
import os
from dotenv import load_dotenv
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader


def test_graph_data_preparation(
    t: int, ont: str, batch_size: int
) -> LinkNeighborLoader:

    load_dotenv()
    root = os.getenv("ROOT", "")

    fp_positive = f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_{ont}_filtered_positive_edges.txt"
    fp_negative = f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_{ont}_filtered_negative_edges.txt"

    if os.path.exists(fp_positive):
        os.remove(fp_positive)
    if os.path.exists(fp_negative):
        os.remove(fp_negative)
    # Load PPI edges
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

    # Create edges for evaluation
    if os.path.exists(
        f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_{ont}_filtered_positive_edges.txt"
    ):
        pos_eval_edges_df = pd.read_csv(
            f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_{ont}_filtered_positive_edges.txt",
            sep="\t",
        )
        pos_eval_edges_df.columns = ["source", "target"]
        neg_eval_edges_df = pd.read_csv(
            f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_{ont}_filtered_negative_edges.txt",
            sep="\t",
        )
        neg_eval_edges_df.columns = ["source", "target"]
    else:
        if ont == "all":
            test_proteins_bp = pd.read_table(
                f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_BP_filtered.txt"
            )["protein_id"].values.tolist()
            test_proteins_mf = pd.read_table(
                f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_MF_filtered.txt"
            )["protein_id"].values.tolist()
            test_proteins_cc = pd.read_table(
                f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_CC_filtered.txt"
            )["protein_id"].values.tolist()
            test_proteins = test_proteins_bp + test_proteins_mf + test_proteins_cc
        else:
            test_proteins = pd.read_table(
                f"{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_{ont}_filtered.txt"
            )["protein_id"].values.tolist()
        positive_edges, negative_edges = [], []
        for protein in test_proteins:
            t1_anno = gt_edges.loc[gt_edges["source"] == protein][
                "target"
            ].values.tolist()
            t2_anno = gt_edges_t2.loc[gt_edges_t2["source"] == protein][
                "target"
            ].values.tolist()
            protein_pe = [
                [protein, anno] for anno in set(t2_anno).difference(set(t1_anno))
            ]
            protein_ne = [
                [protein, anno]
                for anno in t2_terms_list.difference(set(t1_anno + t2_anno))
            ]
            positive_edges.extend(protein_pe)
            negative_edges.extend(protein_ne)
        pos_eval_edges_df = pd.DataFrame(positive_edges, columns=["source", "target"])
        # pos_eval_edges_df.to_csv(f'{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_{ont}_filtered_positive_edges.txt',
        #                          sep='\t', index=False)
        neg_eval_edges_df = pd.DataFrame(negative_edges, columns=["source", "target"])
        # neg_eval_edges_df.to_csv(f'{root}old_code/protein_dataset/benchmark/human_ppi_{t}/test_{ont}_filtered_negative_edges.txt',
        #                          sep='\t', index=False)
    labels_eval = pd.DataFrame(
        [1] * len(pos_eval_edges_df) + [0] * len(neg_eval_edges_df), columns=["label"]
    )
    eval_edges_df = pd.concat((pos_eval_edges_df, neg_eval_edges_df))

    # Create graph
    # G_train = sg.StellarGraph({'protein': protein_nodes_df, 'go_term': go_term_nodes_df},
    #                           {'interaction': ppi_edges, 'annotation': gt_edges_removed})
    # print(G_train.info())

    # graph_data = {
    #     ("go_term", "annotation", "protein"): (gt_edges_removed['source'].values, gt_edges_removed['target'].values),
    #     ("protein", "interaction", "protein"): (ppi_edges['source'].values, ppi_edges['target'].values)
    # }

    protein_map = {protein_nodes_df.index[i]: i for i in range(len(protein_nodes_df))}
    go_term_map = {go_term_nodes_df.index[i]: i for i in range(len(go_term_nodes_df))}

    graph_data = HeteroData()

    # Add graph nodes with their features
    graph_data["protein"].x = torch.tensor(
        protein_nodes_df.values,
    )
    graph_data["go_term"].x = torch.tensor(
        go_term_nodes_df.values,
    )

    ppi_edge_index = [
        [protein_map[p] for p in ppi_edges["source"]],
        [protein_map[p] for p in ppi_edges["target"]],
    ]
    annotation_edge_index = [
        [protein_map[p] for p in gt_edges["source"]],
        [go_term_map[gt] for gt in gt_edges["target"]],
    ]
    eval_edges_index = [
        [protein_map[p] for p in eval_edges_df["source"]],
        [go_term_map[gt] for gt in eval_edges_df["target"]],
    ]

    assert len(ppi_edge_index[0]) == len(ppi_edge_index[1])
    assert len(annotation_edge_index[0]) == len(annotation_edge_index[1])

    # Add graph edges
    graph_data["protein", "interacts", "protein"].edge_index = torch.tensor(
        ppi_edge_index,
    )
    graph_data["protein", "annotated", "go_term"].edge_index = torch.tensor(
        annotation_edge_index,
    )

    # Add graph attributes
    ppi_attributes = torch.tensor(ppi_edges.drop(["source", "target"], axis=1).values)
    graph_data["protein", "interacts", "protein"].edge_attr = ppi_attributes

    # print(graph_data)
    # Create data generator
    # train_generator = FullBatchLinkGenerator(G_train, method='gat')
    # train_generator = HinSAGELinkGenerator(G_train, batch_size, num_samples,
    #                                        head_node_types=['protein', 'go_term'])
    # train_gen = train_generator.flow(edges_to_remove.values, labels_train.values, shuffle=True)
    # assert len(gat_layer_sizes) == len(num_samples)

    graph_data["go_term"].x = graph_data["go_term"].x.to(torch.float32)

    print(graph_data["go_term"].x.dtype)

    test_loader = LinkNeighborLoader(
        data=graph_data,
        num_neighbors=[-1],
        edge_label_index=(
            ("protein", "annotated", "go_term"),
            torch.tensor(
                eval_edges_index,
            ),
        ),
        edge_label=torch.tensor(
            labels_eval.values,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return test_loader
