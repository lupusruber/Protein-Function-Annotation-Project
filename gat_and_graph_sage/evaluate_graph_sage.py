import torch
from tqdm import tqdm
from torch_geometric.data import HeteroData
from hetero_gat_model import GraphSage
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)


def test_model_sage(
    test_loader: HeteroData, device: str, model: GraphSage
) -> dict[str, float]:
    preds = []
    ground_truths = []
    for sampled_data in tqdm(test_loader):
        with torch.no_grad():
            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)
            preds.append(pred)
            ground_truths.append(
                sampled_data["protein", "annotated", "go_term"].edge_label
            )
    pred = torch.cat(preds, dim=0).cpu().numpy()

    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    precision = precision_score(ground_truths, pred.round(), zero_division=1)
    recall = recall_score(ground_truths, pred.round())
    accuracy = accuracy_score(ground_truths, pred.round())
    f1 = f1_score(ground_truths, pred.round())

    res = dict()
    res["test_auc"] = auc
    res["test_precision"] = precision
    res["test_recall"] = recall
    res["test_accuracy"] = accuracy
    res["test_f1"] = f1

    return res
