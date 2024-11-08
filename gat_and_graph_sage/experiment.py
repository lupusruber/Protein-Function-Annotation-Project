from data_preparators.data_preparation_train import train_graph_data_preparation
from tests.data_preparation_test import test_graph_data_preparation
from hetero_gat_model import Model
from train import train
from train_graph_sage import train_sage
from evaluate import test_model
from evaluate_graph_sage import test_model_sage
import torch


def main(
    t: int,
    ont: str,
    batch_size: int,
    gat_hidden_dim: int,
    dropout: float,
    lr: float,
    epochs: int,
) -> None:
    model_name = "GraphSage"
    # model_name = "GAT"
    train_loader, graph_data = train_graph_data_preparation(t, ont, batch_size)
    metadata = graph_data.metadata()

    test_loader = test_graph_data_preparation(t, ont, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "GAT":
        node_dim_protein = next(iter(train_loader))["protein"].x.shape[1]
        node_dim_go_term = next(iter(train_loader))["go_term"].x.shape[1]
        node_dim = {"protein": node_dim_protein, "go_term": node_dim_go_term}
        model = Model(
            model_name=model_name,
            data=graph_data,
            metadata=metadata,
            dim_in=node_dim,
            dim_h=gat_hidden_dim,
            dropout=dropout,
        )
        model, train_result = train(
            model=model,
            train_loader=train_loader,
            batch_size=batch_size,
            epochs=epochs,
            ont=ont,
            t=t,
            log=True,
            save=True,
        )
        test_result = test_model(model, test_loader)
        print(train_result)
        print(test_result)

    elif model_name == "GraphSage":
        model = Model(
            model_name=model_name, metadata=metadata, dim_h=64, data=graph_data
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        graph_data["go_term"].x = graph_data["go_term"].x.to(torch.float32)

        model = train_sage(
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            model=model,
            epochs=3,
        )
        test_result = test_model_sage(
            test_loader=test_loader, device=device, model=model
        )
        print(test_result)


if __name__ == "__main__":
    main(
        t=900,
        ont="CC",
        batch_size=50,
        gat_hidden_dim=256,
        dropout=0.2,
        lr=0.001,
        epochs=1,
    )
