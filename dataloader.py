import csv
import os
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from typing import List, Tuple
import numpy as np


class Config:
    def __init__(self, dataset: str, max_seq_length: int = 50, train_batch_size: int = 32):
        self.dataset = dataset
        self.text_dir = os.path.join(dataset, "text")
        self.audio_file = os.path.normpath(os.path.join(dataset, "audio", "paudio.pickle"))
        self.video_file = os.path.normpath(os.path.join(dataset, "video", "pvideo.pickle"))
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
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
        if not os.path.exists(self.config.audio_file):
            raise FileNotFoundError(f"Audio file not found: {self.config.audio_file}")
        if not os.path.exists(self.config.video_file):
            raise FileNotFoundError(f"Video file not found: {self.config.video_file}")

        with open(self.config.audio_file, "rb") as f:
            train_audio, valid_audio, test_audio = pickle.load(f)

        with open(self.config.video_file, "rb") as f:
            train_video, valid_video, test_video = pickle.load(f)

        train_audio = self._pad_audio_features(train_audio)
        valid_audio = self._pad_audio_features(valid_audio)
        test_audio = self._pad_audio_features(test_audio)

        return train_audio, valid_audio, test_audio, train_video, valid_video, test_video

    def _pad_audio_features(self, audio_data: List[np.ndarray]) -> np.ndarray:
        max_len = max(item.shape[0] for item in audio_data)
        feature_dim = audio_data[0].shape[1]
        padded_data = np.zeros((len(audio_data), max_len, feature_dim), dtype=np.float32)
        for i, item in enumerate(audio_data):
            padded_data[i, :item.shape[0], :] = item
        return padded_data

    def tokenize_text(self, examples: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenizer = self.config.tokenizer
        encoded_inputs = tokenizer(
            examples,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return encoded_inputs["input_ids"], encoded_inputs["attention_mask"], encoded_inputs.get("token_type_ids", None)

    def preprocess_data(self, text_data: List[str], audio_data: np.ndarray, video_data: np.ndarray, labels: List[float]) -> TensorDataset:
        input_ids, attention_masks, token_type_ids = self.tokenize_text(text_data)
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
        video_data = torch.tensor(video_data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        if token_type_ids is None:
            return TensorDataset(input_ids, attention_masks, audio_data, video_data, labels)
        else:
            return TensorDataset(input_ids, attention_masks, token_type_ids, audio_data, video_data, labels)

    def prepare_dataloader(self, data: TensorDataset, batch_size: int, sampler_type: str = "random") -> DataLoader:
        sampler = RandomSampler(data) if sampler_type == "random" else SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=batch_size)


def prepare_mosi_datasets(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    processor = DataPreprocessor(config)

    train_audio, valid_audio, test_audio, train_video, valid_video, test_video = processor.load_features()
    train_texts, train_labels = processor.load_tsv(os.path.join(config.text_dir, "ptrain.tsv"))
    valid_texts, valid_labels = processor.load_tsv(os.path.join(config.text_dir, "pvalid.tsv"))
    test_texts, test_labels = processor.load_tsv(os.path.join(config.text_dir, "ptest.tsv"))

    train_data = processor.preprocess_data(train_texts, train_audio, train_video, train_labels)
    valid_data = processor.preprocess_data(valid_texts, valid_audio, valid_video, valid_labels)
    test_data = processor.preprocess_data(test_texts, test_audio, test_video, test_labels)

    train_loader = processor.prepare_dataloader(train_data, config.train_batch_size, sampler_type="random")
    valid_loader = processor.prepare_dataloader(valid_data, config.train_batch_size, sampler_type="sequential")
    test_loader = processor.prepare_dataloader(test_data, config.train_batch_size, sampler_type="sequential")

    return train_loader, valid_loader, test_loader
