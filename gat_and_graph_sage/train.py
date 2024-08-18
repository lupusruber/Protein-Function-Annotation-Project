from typing import Any
from torch_geometric.loader import LinkNeighborLoader
from torch.nn import BCEWithLogitsLoss

import torch

from hetero_gat_model import GAT
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# train
def train(
    model: GAT,
    train_loader: LinkNeighborLoader,
    batch_size: int,
    epochs: int,
    ont: str,
    t: int,
    log: bool = False,
    save: bool = False,
) -> tuple[GAT, dict[Any, Any]]:
    criterion = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    res = dict()

    model.train()
    for epoch in range(epochs):
        losses = []
        preds = []
        ground_truths = []

        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch)
            ground_truth = batch["protein", "annotated", "go_term"].edge_label.to(
                torch.float32
            )
            pred = pred.reshape_as(ground_truth)
            # pred = pred.round()
            preds.append(pred)
            ground_truths.append(ground_truth)

            loss = criterion(pred, ground_truth)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().numpy())

        epoch_loss = sum(losses) / len(losses) if len(losses) > 0 else 0
        preds_tensor: torch.Tensor = torch.cat(preds, dim=0).detach().numpy()
        ground_truths = torch.cat(ground_truths, dim=0).detach().numpy()
        auc = roc_auc_score(ground_truths, preds_tensor)
        precision = precision_score(
            ground_truths, preds_tensor.round(), zero_division=1
        )
        recall = recall_score(ground_truths, preds_tensor.round())
        accuracy = accuracy_score(ground_truths, preds_tensor.round())
        f1 = f1_score(ground_truths, preds_tensor.round())
        res["epoch"] = epoch
        res["epoch_loss"] = epoch_loss
        res["auc"] = auc
        res["precision"] = precision
        res["recall"] = recall
        res["f1_score"] = f1
        res["accuracy"] = accuracy
        print(f"Epoch: {epoch}, loss {epoch_loss}, auc: {auc}")

        # if save and (epoch+1) % 20 == 0:
        #     path_to_dir = f"gat\checkpoints\{ont}_{t}"
        #     if not os.path.exists(path_to_dir):
        #         os.mkdir(path_to_dir)
        #     torch.save(model.state_dict(), f"gat\checkpoints\{ont}_{t}\checkpoint_{epoch+1}")

    return model, res
