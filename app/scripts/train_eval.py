import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from constants import TWITTER_GRAPH_PATH, TWITTER_GRAPH_MODEL_PATH, device
from app.models.graph_models import GCN, GAT
from app.core.graph_related.graph import create_custom_shot_train_mask


# Training function
def train(model, optimizer, loss_fn, graph_data):
    """Train the GCN model on the graph data.
    Args:
        model: The GCN model to train.
        optimizer: The optimizer for training.
        loss_fn: The loss function for training.
        graph_data: The graph data containing node features, edge indices, and masks.
    Returns:
        loss (float): The loss value after training.
    """
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
    """Evaluate the model on validation or test set.
    Args:
        graph_data: The graph data containing node features, edge indices, and masks.
        model: The GCN model to evaluate.
        test (bool): If True, evaluate on the test set; otherwise, evaluate on the validation set.
    Returns:
        acc (float): Accuracy of the model on the validation or test set.
        precision (float): Precision of the model on the validation or test set.
        recall (float): Recall of the model on the validation or test set.
        f1 (float): F1 score of the model on the validation or test set.
    """
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


def train_eval_model(
    data_type: str,
    model_type: str = "gcn",
    epochs: int = 200,
    learning_rate: float = 0.01,
    few_shot: bool = False,
    number_of_shots: int = 1,
):
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
            os.path.join(TWITTER_GRAPH_PATH, f"graph_{data_type}_splitted.pt"),
            map_location=torch.device(device),
        )
        if graph_data is None:
            print(
                "Graph data is None. Please check if the graph was created and split correctly."
            )
            return None
        # Create one-shot train mask if needed
        graph_data = (
            create_custom_shot_train_mask(graph_data, number_of_shots)
            if few_shot
            else graph_data
        )
        input_dim = graph_data.num_node_features
        model = (
            GAT(input_dim, hidden_dim).to(device)
            if model_type == "gat"
            else GCN(input_dim, hidden_dim, output_dim).to(device)
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=5e-4
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary classification loss
        best_val_acc = 0
        patience = 100  # Stop if validation accuracy does not improve for 10 epochs
        wait = 0
        saved_model_path = os.path.join(
            TWITTER_GRAPH_MODEL_PATH, f"{model_type}_{data_type}_model.pt"
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
