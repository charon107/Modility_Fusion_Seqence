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
import logging
from logging import Logger
from sklearn.metrics import mean_squared_error

# 配置基础日志设置
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

# 创建全局日志记录器
logger: Logger = logging.getLogger(__name__)

def setup_optimizer_scheduler(model, total_steps):
    """适配新版模型结构的优化器设置"""
    # 更精细的参数分组策略
    param_groups = [
        {  # 文本编码器参数
            "params": model.text_encoder.parameters(),
            "lr": 1e-5,
            "weight_decay": 0.01
        },
        {  # 视觉编码器参数
            "params": model.video_encoder.parameters(),
            "lr": 5e-5,
            "weight_decay": 0.005
        },
        {  # 音频编码器参数
            "params": model.audio_encoder.parameters(),
            "lr": 5e-5,
            "weight_decay": 0.005
        },
        {  # 跨模态注意力参数
            "params": model.text_audio_attn.parameters(),
            "lr": 1e-4,
            "weight_decay": 0.001
        },
        {  # 分类器参数
            "params": model.classifier.parameters(),
            "lr": 2e-4,
            "weight_decay": 0.001
        }
    ]

    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)

    # 分阶段学习率调度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in param_groups],
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    return optimizer, scheduler, torch.nn.SmoothL1Loss()  # 改用Huber损失


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device,
                    gradient_accumulation=4, max_grad_norm=1.0):
    """适配新模型前向传播的训练步骤"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        with torch.amp.autocast(device_type='cuda', enabled=True):  # 更新autocast语法
            # 解包批次数据
            text_inputs, text_mask, _, audio, video, labels = [t.to(device) for t in batch]

            # 检查输入数据是否包含NaN或Inf
            if torch.isnan(text_inputs).any() or torch.isinf(text_inputs).any():
                raise ValueError("NaN or Inf detected in text inputs")
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                raise ValueError("NaN or Inf detected in audio inputs")
            if torch.isnan(video).any() or torch.isinf(video).any():
                raise ValueError("NaN or Inf detected in video inputs")

            # 前向传播适配新模型接口
            outputs = model(
                text_inputs=text_inputs,
                text_masks=text_mask,
                audio_inputs=audio,
                video_inputs=video
            )

            # 检查输出是否包含NaN或Inf
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
    os.makedirs(output_dir, exist_ok=True)

    # 调整总步数计算
    total_steps = num_epochs * (len(train_loader) // train_loader.batch_size)

    optimizer, scheduler, criterion = setup_optimizer_scheduler(model, total_steps)
    scaler = GradScaler(device_type='cuda', enabled=True)

    # 精简模型初始化
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        logger.warning("Detected multiple GPUs but running in single-GPU mode")

    # 轻量级监控数据
    metrics = {
        'train_loss': [],
        'val_pearson': [],
        'val_mse': []
    }

    best_pearson = -1.0

    # 训练循环优化
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()

        # 训练阶段
        model.train()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, gradient_accumulation=8  # 8步梯度累积
        )
        scheduler.step()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            preds, labels = [], []
            for batch in val_loader:
                batch = [t.to(device, non_blocking=True) for t in batch]
                text_inputs, text_mask, _, audio, video, label = batch

                outputs = model(text_inputs, text_mask, audio, video)
                preds.append(outputs.squeeze().cpu().float().numpy())
                labels.append(label.cpu().float().numpy())

                # 及时释放中间变量
                del outputs
                torch.cuda.empty_cache()

        # 计算指标
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        mse = mean_squared_error(labels, preds)
        pearson = pearsonr(labels, preds)[0]

        # 保存最佳模型
        if pearson > best_pearson:
            best_pearson = pearson
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info(f"New best model saved with Pearson {pearson:.4f}")

        # 记录指标
        metrics['train_loss'].append(avg_loss)
        metrics['val_pearson'].append(pearson)
        metrics['val_mse'].append(mse)

        # 保存轻量级检查点
        if (epoch + 1) % 3 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(output_dir, f"checkpoint_epoch{epoch + 1}.pt"))

        # 打印精简日志
        logger.info(
            f"Epoch {epoch + 1} | "
            f"Loss: {avg_loss:.4f} | "
            f"Pearson: {pearson:.4f} | "
            f"MSE: {mse:.4f} | "
            f"Time: {time.time() - epoch_start:.1f}s"
        )

        # 每3个epoch清理显存
        if (epoch + 1) % 3 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # 可视化优化
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
    # 创建配置对象时设置所有参数
    config = Config(
        dataset="D:/Project/Modility_Fusion_Seqence/Data/MOSEI",
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        max_seq_length=50
    )

    # 确保prepare_mosi_datasets返回三个数据加载器
    try:
        train_loader, val_loader, test_loader = prepare_mosi_datasets(config)
        print(f"Data loaders created: Train={train_loader}, Val={val_loader}, Test={test_loader}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        exit(1)

    # 初始化模型（根据实际模型架构调整参数）
    model = MultimodalFusionNetwork(
        text_model="bert-base-uncased",
        audio_feature_dim=1582,
        video_feature_dim=711,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    )

    # 启动训练
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

    # 最终清理
    del trained_model
    gc.collect()
    torch.cuda.empty_cache()