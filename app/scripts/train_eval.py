import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from constants import GRAPH_PATH, GRAPH_MODEL_PATH, device
from app.models.graph_models import GCN, GAT


# Training function
def train(model, optimizer, loss_fn, graph_data):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = loss_fn(
        out[graph_data.train_mask],
        graph_data.y[graph_data.train_mask].squeeze().float(),
    )
    loss.backward()
    optimizer.step()
    return loss.item()


# Evaluation function
def evaluate(graph_data, model, test=False):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation

        # Forward pass
        logits = model(graph_data.x, graph_data.edge_index)

        # Convert logits to probabilities using sigmoid
        probs = torch.sigmoid(logits)

        # Convert probabilities to binary predictions (0 or 1)
        preds = (probs > 0.5).cpu().numpy()

        # Evaluate for validation or test set
        preds = (
            preds[graph_data.test_mask.cpu().numpy()]
            if test
            else preds[graph_data.val_mask.cpu().numpy()]
        )

        y_true = (
            graph_data.y[graph_data.test_mask].squeeze().cpu().numpy()
            if test
            else graph_data.y[graph_data.val_mask].squeeze().cpu().numpy()
        )
        # Compute accuracy
        acc = accuracy_score(y_true, preds)
        precision = precision_score(y_true, preds, average="binary")
        recall = recall_score(y_true, preds, average="binary")
        f1 = f1_score(y_true, preds, average="binary")
        return acc, precision, recall, f1


def train_eval_model(data_type: str, model_type: str = "gcn", epochs: int = 200, learning_rate: float = 0.01):
    """
    Train and evaluate the GCN model.
    Args:
        data_type (str): type of graph (twitter, geotext).
        model_type (str): type of model to use (currently "gcn" and "gat" is supported).
    """
    try:
        if model_type not in ["gcn", "gat"]:
            print(
                "Invalid model type. Supported types are 'gcn' and 'gat'. Defaulting to 'gcn'."
            )
        hidden_dim = 32
        output_dim = 2 if data_type == "geotext" else 1
        graph_data = torch.load(
            os.path.join(GRAPH_PATH, f"graph_{data_type}_splitted.pt"),
            map_location=torch.device(device),
        )
        if graph_data is None:
            print(
                "Graph data is None. Please check if the graph was created and split correctly."
            )
            return None
        input_dim = graph_data.num_node_features
        model = (
            GAT(input_dim, hidden_dim).to(device)
            if model_type == "gat"
            else GCN(input_dim, hidden_dim, output_dim).to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary classification loss
        best_val_acc = 0
        patience = 50  # Stop if validation accuracy does not improve for 10 epochs
        wait = 0
        saved_model_path = os.path.join(
            GRAPH_MODEL_PATH, f"{model_type}_{data_type}_model.pt"
        )
        for epoch in range(epochs):
            loss = train(model, optimizer, loss_fn, graph_data)
            val_acc, _, _, _ = evaluate(graph_data, model)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(
                        f"Early stopping at epoch {epoch}. Best validation accuracy: {best_val_acc:.4f}"
                    )
                    break

            if epoch % 20 == 0:
                result = evaluate(graph_data, model, test=True)
                print(
                    f"Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {result[0]:.4f}"
                )

        # Final evaluation
        acc, precision, recall, f1 = evaluate(graph_data, model, test=True)
        print(f"Final Test Accuracy: {acc:.4f}")
        print(f"Final Test Precision: {precision:.4f}")
        print(f"Final Test Recall: {recall:.4f}")
        print(f"Final Test F1: {f1:.4f}")
        torch.save(
            model.state_dict(),
            saved_model_path,
        )
        print(f"Model saved to {saved_model_path}.")
    except Exception as e:
        print(f"An error occurred during training/evaluation: {e}")
        return None
