import csv
import os
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from typing import List, Tuple
import numpy as np


class Config:
    def __init__(self,
                 dataset: str,
                 batch_size: int = 8,
                 num_workers: int = 2,
                 max_seq_length: int = 50,
                 pin_memory: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.pin_memory = pin_memory

        # 其他路径参数保持不变
        self.text_dir = os.path.join(dataset, "text")
        self.audio_file = os.path.normpath(os.path.join(dataset, "audio", "paudio.pickle"))
        self.video_file = os.path.normpath(os.path.join(dataset, "video", "pvideo.pickle"))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config

    def load_tsv(self, file_path: str) -> Tuple[List[str], List[float]]:
        texts, labels = [], []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                texts.append(row[0])
                labels.append(float(row[1]))
        return texts, labels

    def load_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 添加文件存在性检查
        for path in [self.config.audio_file, self.config.video_file]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Feature file not found: {path}")

        with open(self.config.audio_file, "rb") as f:
            train_audio, valid_audio, test_audio = pickle.load(f)

        with open(self.config.video_file, "rb") as f:
            train_video, valid_video, test_video = pickle.load(f)

        return (
            self._pad_features(train_audio),
            self._pad_features(valid_audio),
            self._pad_features(test_audio),
            self._pad_features(train_video),
            self._pad_features(valid_video),
            self._pad_features(test_video)
        )

    def _pad_features(self, features: List[np.ndarray]) -> np.ndarray:
        """统一处理音频和视频特征的填充"""
        max_len = max(item.shape[0] for item in features)
        feature_dim = features[0].shape[1]
        padded = np.zeros((len(features), max_len, feature_dim), dtype=np.float32)
        for i, item in enumerate(features):
            padded[i, :item.shape[0], :] = item
        return padded

    def tokenize_text(self, examples: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenizer = self.config.tokenizer
        return tokenizer(
            examples,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True  # 确保返回token_type_ids
        ).values()

    def preprocess_data(self,
                        text_data: List[str],
                        audio_data: np.ndarray,
                        video_data: np.ndarray,
                        labels: List[float]) -> TensorDataset:
        # 解包tokenizer输出
        input_ids, attention_mask, token_type_ids = self.tokenize_text(text_data)

        return TensorDataset(
            input_ids,
            attention_mask,
            token_type_ids,
            torch.tensor(audio_data, dtype=torch.float32),
            torch.tensor(video_data, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32)
        )

    def prepare_dataloader(self, dataset: TensorDataset, is_training: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,  # 使用config中的参数
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            sampler=RandomSampler(dataset) if is_training else SequentialSampler(dataset)
        )


def prepare_mosi_datasets(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    processor = DataPreprocessor(config)

    # 加载特征数据
    (train_audio, valid_audio, test_audio,
     train_video, valid_video, test_video) = processor.load_features()

    # 加载文本和标签
    def load_split(split: str):
        texts, labels = processor.load_tsv(
            os.path.join(config.text_dir, f"p{split}.tsv")
        )
        return texts, labels

    train_texts, train_labels = load_split("train")
    valid_texts, valid_labels = load_split("valid")
    test_texts, test_labels = load_split("test")

    # 创建数据集
    train_data = processor.preprocess_data(train_texts, train_audio, train_video, train_labels)
    valid_data = processor.preprocess_data(valid_texts, valid_audio, valid_video, valid_labels)
    test_data = processor.preprocess_data(test_texts, test_audio, test_video, test_labels)

    # 创建数据加载器
    return (
        processor.prepare_dataloader(train_data, is_training=True),
        processor.prepare_dataloader(valid_data, is_training=False),
        processor.prepare_dataloader(test_data, is_training=False)
    )