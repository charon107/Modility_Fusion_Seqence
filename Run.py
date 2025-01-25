import os
import time
import torch
import torch.optim as optim
import logging
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataloader import DataPreprocessor, Config, prepare_mosi_datasets
from model import MultimodalFusion
from evaluate import evaluate
import gc
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import random
import json
from transformers import AutoModelForSequenceClassification
import matplotlib.pyplot as plt  # Import matplotlib for plotting loss curves

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU random seed
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Avoid non-deterministic algorithms

# Setup logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed before starting training
set_seed(42)  # You can change the seed value if needed

def setup_optimizer_scheduler(model):
    """
    Setup optimizer, scheduler, and loss function for training.
    """
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = torch.nn.MSELoss()
    return optimizer, scheduler, criterion

def save_best_model(model, eval_accuracy, best_eval_accuracy, epoch, output_dir):
    """
    Save the model if validation accuracy improves. Only save the model weights (no config file).
    """
    if eval_accuracy[0] > best_eval_accuracy:
        best_eval_accuracy = eval_accuracy[0]  # Access the first element (accuracy)
        output_model_dir = os.path.join(output_dir, f"best_model_epoch_{epoch + 1}")
        os.makedirs(output_model_dir, exist_ok=True)
        logger.info(f"Saving model with accuracy {eval_accuracy[0]:.4f} to {output_model_dir}")  # Access the accuracy value

        # Save model weights (no config file needed for custom model)
        torch.save(model.state_dict(), os.path.join(output_model_dir, "pytorch_model.bin"))

    return best_eval_accuracy


def train_one_epoch(model, train_dataloader, optimizer, scheduler, criterion, scaler, device,
                    gradient_accumulation_steps=1, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training", unit="batch")):
        batch = [item.to(device) for item in batch]
        if len(batch) == 5:
            text_inputs, text_masks, audio_inputs, video_inputs, labels = batch
        elif len(batch) == 6:
            text_inputs, text_masks, token_type_ids, audio_inputs, video_inputs, labels = batch
        else:
            raise ValueError(f"Unexpected number of items in batch: {len(batch)}")

        with autocast():
            outputs = model(text_inputs=text_inputs, text_masks=text_masks, audio_inputs=audio_inputs, video_inputs=video_inputs)
            loss = criterion(outputs.squeeze(-1), labels)

        total_loss += loss.item()
        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def load_model(model_path, device, model_class):
    """
    Load a pre-trained custom model from the given path, including its weights.

    Args:
        model_path: Path to the model directory.
        device: The device to load the model on (cuda or cpu).
        model_class: The class of your custom model.

    Returns:
        model: The loaded model.
    """
    # Initialize the model using the custom model class
    model = model_class()

    # Load the model weights from the pytorch_model.bin
    model_weights_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model weights 'pytorch_model.bin' not found at {model_path}")

    model.to(device)  # Ensure the model is moved to the correct device (CPU or GPU)
    return model


def evaluate_model(model, val_dataloader, device, eval_type="classification"):
    """
    Evaluate the model on the validation dataset.

    Args:
        model: The trained model.
        val_dataloader: The dataloader for the validation dataset.
        device: The device to run the evaluation on (cuda or cpu).
        eval_type: Type of evaluation ("classification" or "regression").

    Returns:
        accuracy: The accuracy score for classification tasks.
        f1: The F1 score for classification tasks.
        mse: The Mean Squared Error for regression tasks.
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            text_inputs, text_masks, token_type_ids, audio_inputs, video_inputs, labels = batch
            text_inputs = text_inputs.to(device)
            text_masks = text_masks.to(device)
            audio_inputs = audio_inputs.to(device)
            video_inputs = video_inputs.to(device)
            labels = labels.to(device)

            # Get model predictions
            outputs = model(text_inputs, text_masks, audio_inputs, video_inputs)

            # For regression tasks, directly take the outputs as predictions
            if eval_type == "regression":
                preds = outputs.squeeze().cpu().numpy()  # Remove extra dimensions
            else:
                # For classification tasks, use threshold to classify the predictions
                preds = (outputs > 0.5).float().cpu().numpy()  # Apply threshold for binary classification

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    if eval_type == "classification":
        all_preds = (np.array(all_preds) > 0.5).astype(int)  # Convert to binary labels
        all_labels = np.array(all_labels).astype(int)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        mse = None  # MSE is not needed for classification tasks
    else:
        accuracy = None  # Accuracy is not used for regression tasks
        f1 = None  # F1 score is not used for regression tasks
        mse = mean_squared_error(all_labels, all_preds)  # MSE for regression tasks

    print(f"Evaluation Results: Accuracy: {accuracy}, F1: {f1}, MSE: {mse}")

    return accuracy, f1, mse

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs=10,
                gradient_accumulation_steps=1, output_dir='./output', device='cuda', max_grad_norm=1.0):
    """
    Train and evaluate the model, including saving the best model based on validation accuracy.
    """
    model.to(device)
    scaler = GradScaler()
    train_losses = []  # List to store the training loss for plotting

    # If using multi-GPU setup
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    best_eval_accuracy = 0.0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Train the model for one epoch
        start_time = time.time()
        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, criterion, scaler, device,
                                         gradient_accumulation_steps, max_grad_norm)

        # Log training information
        logger.info(f"Train Loss: {avg_train_loss:.4f} - Time: {time.time() - start_time:.2f}s")

        # Append the training loss for plotting
        train_losses.append(avg_train_loss)

        # Evaluate the model after each epoch (using the trained model)
        eval_accuracy = evaluate_model(model, val_dataloader, device)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {eval_accuracy[0]:.4f}")

        # Save the model if the validation accuracy improves
        best_eval_accuracy = save_best_model(model, eval_accuracy, best_eval_accuracy, epoch, output_dir)

    # Plot the loss curve after all epochs are done
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    logger.info("Training completed.")

if __name__ == "__main__":
    # Set configuration paths and parameters
    config = Config(dataset="D:/Project/Modility_Fusion_Seqence/Data/MOSEI")

    # Prepare data (train, validation, test loaders)
    processor = DataPreprocessor(config)
    train_loader, valid_loader, test_loader = prepare_mosi_datasets(config)

    # Initialize model
    model = MultimodalFusion(embed_dim=256, num_heads=8, num_layers=4, dropout=0.1, output_dim=1)

    # Setup optimizer, scheduler, and loss function
    optimizer, scheduler, criterion = setup_optimizer_scheduler(model)

    # Train the model
    train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, num_epochs=12,
                gradient_accumulation_steps=2, output_dir='./output', device='cuda')

    # Clean up GPU memory
    gc.collect()
