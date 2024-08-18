import torch

from torch_geometric.loader import LinkNeighborLoader
from hetero_gat_model import GAT
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)


def test_model(model: GAT, test_loader: LinkNeighborLoader) -> dict[str, float]:
    model.eval()
    preds = []
    ground_truths = []

    for batch in test_loader:
        with torch.no_grad():
            pred = model(batch)
            ground_truth = batch["protein", "annotated", "go_term"].edge_label.to(
                torch.float32
            )
            pred = pred.reshape_as(ground_truth)

            preds.append(pred)
            ground_truths.append(ground_truth)

    res = dict()
    preds_tensor: torch.Tensor = torch.cat(preds, dim=0).detach().numpy()
    ground_truths = torch.cat(ground_truths, dim=0).detach().numpy()
    auc = roc_auc_score(ground_truths, preds_tensor)
    precision = precision_score(ground_truths, preds_tensor.round(), zero_division=1)
    recall = recall_score(ground_truths, preds_tensor.round())
    accuracy = accuracy_score(ground_truths, preds_tensor.round())
    f1 = f1_score(ground_truths, preds_tensor.round())

    res["test_auc"] = auc
    res["test_precision"] = precision
    res["test_recall"] = recall
    res["test_accuracy"] = accuracy
    res["test_f1"] = f1

    return res
