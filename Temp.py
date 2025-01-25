import torch
from dataloader import Config, prepare_mosi_datasets
from model import MultimodalFusion
from torch.utils.data import DataLoader

def main():
    # 配置文件
    dataset_path = "D:/Project/Modility_Fusion_Seqence/Data/MOSI"  # 请根据实际路径修改
    config = Config(dataset=dataset_path)

    # 加载数据集
    train_loader, valid_loader, test_loader = prepare_mosi_datasets(config)

    # 初始化模型
    embed_dim = 256  # 你可以根据需要调整
    num_heads = 4    # 你可以根据需要调整
    num_layers = 2   # 你可以根据需要调整
    model = MultimodalFusion(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)

    # 选择设备：GPU或CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 以验证集为例进行前向传播（训练和测试集类似）
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for batch in train_loader:
            # 从dataloader中取出数据
            text_inputs, text_masks, token_type_ids, audio_inputs, video_inputs, labels = batch
            text_inputs = text_inputs.to(device)
            text_masks = text_masks.to(device)
            audio_inputs = audio_inputs.to(device)
            video_inputs = video_inputs.to(device)
            # print(text_inputs.shape)
            # print(text_masks.shape)
            # print(audio_inputs.shape)
            # print(video_inputs.shape)

            # 前向传播
            roberta_features = model(text_inputs, text_masks, audio_inputs, video_inputs)

            # 输出
            print("Roberta features shape:", roberta_features.shape)

            # 可以选择返回或保存结果等

            # 为了演示，这里只取一轮数据
            break

if __name__ == "__main__":
    main()
