import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


def accuracy_7(preds, labels):
    """
    Calculate the accuracy based on whether both the predictions and labels
    are either positive or negative.

    Args:
        preds: The predictions made by the model.
        labels: The true labels.

    Returns:
        accuracy: The accuracy score.
    """
    num = 0
    for i in range(len(preds)):
        if preds[i] > 0 and labels[i] > 0:
            num += 1
        elif preds[i] < 0 and labels[i] < 0:
            num += 1
    return num / float(len(preds))


def accuracy(preds, labels):
    """
    Calculate the accuracy based on a threshold of 0.5 for binary classification.

    Args:
        preds: The predictions made by the model.
        labels: The true labels.

    Returns:
        accuracy: The accuracy score.
    """
    preds = np.array(preds)
    labels = np.array(labels)
    return np.sum((preds >= 0.5) == (labels >= 0.5)) / float(len(labels))


def f1_score_custom(preds, labels):
    """
    Calculate the F1 score for multiclass classification.

    Args:
        preds: The predictions made by the model.
        labels: The true labels.

    Returns:
        f1: The F1 score.
    """
    preds = np.array(preds)
    labels = np.array(labels)
    # Use 'macro' or 'weighted' for multiclass classification
    return f1_score(labels, preds, average='macro')



def pearson(preds, labels):
    """
    Calculate the Pearson correlation coefficient for regression tasks.

    Args:
        preds: The predictions made by the model.
        labels: The true labels.

    Returns:
        pearson_corr: The Pearson correlation coefficient.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    # Calculate the Pearson correlation coefficient
    sum_preds = np.sum(preds)
    sum_labels = np.sum(labels)
    sum_preds_squared = np.sum(preds ** 2)
    sum_labels_squared = np.sum(labels ** 2)
    sum_product = np.sum(preds * labels)

    num = sum_product - (sum_preds * sum_labels / len(preds))
    den = np.sqrt(
        (sum_preds_squared - (sum_preds ** 2) / len(preds)) * (sum_labels_squared - (sum_labels ** 2) / len(labels)))

    if den == 0:
        return 0.0
    return num / den


def evaluate(preds, labels, device, eval_type="classification"):
    """
    Evaluate the model using predictions and ground truth labels for classification or regression tasks.

    Args:
        preds: The predictions made by the model.
        labels: The true labels.
        device: The device to run the evaluation on (cuda or cpu).
        eval_type: Type of evaluation ("classification" or "regression").

    Returns:
        accuracy: The accuracy score for classification tasks.
        f1: The F1 score for classification tasks.
        mse: The Mean Squared Error for regression tasks.
    """
    # Move data to device
    preds = torch.tensor(preds).to(device)
    labels = torch.tensor(labels).to(device)

    if eval_type == "classification":
        # For binary classification, apply threshold to predictions
        pred_labels = (preds >= 0.5).float()

        # Accuracy calculation
        accuracy_val = accuracy(preds.cpu().numpy(), labels.cpu().numpy())

        # F1 score calculation
        f1_val = f1_score_custom(preds.cpu().numpy(), labels.cpu().numpy())

        # For regression, mean squared error (MSE) calculation
        mse_val = mean_squared_error(labels.cpu().numpy(), preds.cpu().numpy())

        return accuracy_val, f1_val, mse_val
    else:
        # For regression tasks, calculate Pearson correlation and MSE
        pearson_corr = pearson(preds.cpu().numpy(), labels.cpu().numpy())
        mse_val = mean_squared_error(labels.cpu().numpy(), preds.cpu().numpy())

        return pearson_corr, mse_val
