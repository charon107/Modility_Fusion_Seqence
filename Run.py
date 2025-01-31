import os
import time
import torch
import torch.optim as optim
import logging
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataloader import DataPreprocessor, Config, prepare_mosi_datasets
from model import MultimodalFusionNetwork
import gc
import random
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Set up logging configuration
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

# Create global logger
logger: logging.Logger = logging.getLogger(__name__)

def setup_optimizer_scheduler(model, total_steps):
    """
    Setup optimizer and scheduler for model training.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    total_steps : int
        The total number of training steps.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The optimizer for training.
    scheduler : torch.optim.lr_scheduler.OneCycleLR
        The learning rate scheduler.
    criterion : torch.nn.Module
        The loss function.
    """
    param_groups = [
        {  # Text encoder parameters
            "params": model.text_encoder.parameters(),
            "lr": 5e-6,
            "weight_decay": 0.01
        },
        {  # Video encoder parameters
            "params": model.video_encoder.parameters(),
            "lr": 5e-5,
            "weight_decay": 0.005
        },
        {  # Audio encoder parameters
            "params": model.audio_encoder.parameters(),
            "lr": 2.5e-5,
            "weight_decay": 0.005
        },
        {  # Cross-modal attention parameters
            "params": model.text_audio_attn.parameters(),
            "lr": 5e-5,
            "weight_decay": 0.001
        },
        {  # Classifier parameters
            "params": model.classifier.parameters(),
            "lr": 1e-4,
            "weight_decay": 0.001
        }
    ]

    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-6,  # 更严格的epsilon
        weight_decay=0.01
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in param_groups],
        total_steps=total_steps,
        pct_start=0.2,
        div_factor=25,  # 初始学习率 = max_lr / 25
        final_div_factor=1e4
    )

    return optimizer, scheduler, torch.nn.SmoothL1Loss()


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device,
                    gradient_accumulation=4, max_grad_norm=0.5):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    dataloader : torch.utils.data.DataLoader
        The DataLoader for the training dataset.
    optimizer : torch.optim.Optimizer
        The optimizer for training.
    criterion : torch.nn.Module
        The loss function.
    scaler : torch.amp.GradScaler
        The gradient scaler for mixed precision training.
    device : torch.device
        The device to run the model on.
    gradient_accumulation : int, optional
        The number of gradient accumulation steps, by default 4.
    max_grad_norm : float, optional
        The maximum gradient norm for gradient clipping, by default 0.5.

    Returns
    -------
    float
        The average loss for the epoch.
    """
    with torch.amp.autocast(device_type='cuda', enabled=True):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, desc="Training")):
            with torch.amp.autocast(device_type='cuda', enabled=True):
                # Unpack batch data
                text_inputs, text_mask, _, audio, video, labels = [t.to(device) for t in batch]

                # Forward pass for the model
                outputs = model(
                    text_inputs=text_inputs,
                    text_masks=text_mask,
                    audio_inputs=audio,
                    video_inputs=video
                )

                # Check for NaN or Inf in outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    raise ValueError("NaN or Inf detected in model outputs")

                loss = criterion(outputs.squeeze(), labels) / gradient_accumulation

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    raise ValueError("NaN or Inf detected in gradients")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            total_loss += loss.item() * gradient_accumulation


        return total_loss / len(dataloader)



def train_model(model, train_loader, val_loader, num_epochs=12,
                output_dir="./output", device="cuda"):
    """
    Train the model for a number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        The DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        The DataLoader for the validation dataset.
    num_epochs : int, optional
        The number of epochs to train the model, by default 12.
    output_dir : str, optional
        The directory to save model checkpoints, by default './output'.
    device : str, optional
        The device to run the model on, by default 'cuda'.

    Returns
    -------
    torch.nn.Module
        The trained model.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Adjust total steps calculation
    total_steps = num_epochs * (len(train_loader) // train_loader.batch_size)

    optimizer, scheduler, criterion = setup_optimizer_scheduler(model, total_steps)
    scaler = GradScaler(device='cuda', enabled=True)

    # Move model to the specified device
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        logger.warning("Detected multiple GPUs but running in single-GPU mode")

    # Initialize metrics for tracking
    metrics = {
        'train_loss': [],
        'val_pearson': [],
        'val_mse': []
    }

    best_pearson = -1.0

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()

        # Training phase
        model.train()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, gradient_accumulation=8  # 8-step gradient accumulation
        )
        scheduler.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            preds, labels = [], []
            for batch in val_loader:
                batch = [t.to(device, non_blocking=True) for t in batch]
                text_inputs, text_mask, _, audio, video, label = batch

                outputs = model(text_inputs, text_mask, audio, video)
                preds.append(outputs.squeeze().cpu().float().numpy())
                labels.append(label.cpu().float().numpy())

                # Release intermediate variables
                del outputs
                torch.cuda.empty_cache()

        # Compute metrics
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        mse = mean_squared_error(labels, preds)
        pearson = pearsonr(labels, preds)[0]

        # Save the best model based on Pearson correlation
        if pearson > best_pearson:
            best_pearson = pearson
            torch.save(model.state_dict(), os.path.join(output_dir, "Roberta_TAV.pt"))
            logger.info(f"New best model saved with Pearson {pearson:.4f}")

        # Log metrics
        metrics['train_loss'].append(avg_loss)
        metrics['val_pearson'].append(pearson)
        metrics['val_mse'].append(mse)

        # Save checkpoints every 3 epochs
        if (epoch + 1) % 3 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(output_dir, f"checkpoint_epoch{epoch + 1}.pt"))

        # Print summarized logs
        logger.info(
            f"Epoch {epoch + 1} | "
            f"Loss: {avg_loss:.4f} | "
            f"Pearson: {pearson:.4f} | "
            f"MSE: {mse:.4f} | "
            f"Time: {time.time() - epoch_start:.1f}s"
        )

        # Free memory every 3 epochs
        if (epoch + 1) % 3 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Visualize training metrics
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_loss'], label='Training Loss')
    plt.plot(metrics['val_mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), bbox_inches='tight')
    plt.close()

    logger.info("Training completed")
    return model


if __name__ == "__main__":
    # Set up configuration
    config = Config(
        dataset="D:/Project/Modility_Fusion_Seqence/Data/MOSEI",
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        max_seq_length=50
    )

    # Ensure that prepare_mosi_datasets returns three data loaders
    try:
        train_loader, val_loader, test_loader = prepare_mosi_datasets(config)
        print(f"Data loaders created: Train={train_loader}, Val={val_loader}, Test={test_loader}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        exit(1)

    # Initialize the model
    model = MultimodalFusionNetwork(
        text_model="bert-base-uncased",
        audio_feature_dim=1582,
        video_feature_dim=711,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    )

    # Start training
    try:
        trained_model = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=15,
            output_dir="./output",
            device="cuda"
        )
    except Exception as e:
        print(f"Training failed: {e}")
        exit(1)
