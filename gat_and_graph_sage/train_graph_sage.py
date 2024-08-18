import torch
import torch.nn.functional as F
from tqdm import tqdm
from hetero_gat_model import GraphSage
from torch_geometric.loader import LinkNeighborLoader


def train_sage(
    train_loader: LinkNeighborLoader,
    optimizer: torch.optim.Adam,
    device: torch.device,
    model: GraphSage,
    epochs: int,
) -> GraphSage:
    model.train()
    for epoch in range(epochs):
        total_loss = total_examples = 0
        for sampled_data in tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data[
                "protein", "annotated", "go_term"
            ].edge_label.squeeze()
            ground_truth = ground_truth.to(torch.float32)
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    return model
