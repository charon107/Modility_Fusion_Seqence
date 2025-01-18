import csv
import os
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from typing import List, Tuple
import numpy as np


class Config:
    """
    Configuration for dataset paths and preprocessing parameters.

    Attributes
    ----------
    dataset : str
        The name of the dataset (e.g., "MOSI").
    text_dir : str
        The directory containing text data.
    audio_file : str
        Path to the file containing audio features.
    audio_is_file : str
        Path to the file containing audio-IS features.
    video_file : str
        Path to the file containing video features.
    max_seq_length : int
        The maximum length of the tokenized input sequences.
    train_batch_size : int
        Batch size for training.
    tokenizer : BertTokenizer
        The tokenizer for text preprocessing.
    """
    def __init__(self, dataset: str, max_seq_length: int = 50, train_batch_size: int = 32):
        self.dataset = dataset
        self.text_dir = os.path.join(dataset, "text")
        self.audio_file = os.path.join(dataset, "audio", "paudio.pickle")
        self.audio_is_file = os.path.join(dataset, "audio", "paudio_IS.pickle")
        self.video_file = os.path.join(dataset, "video", "pvideo.pickle")
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class DataPreprocessor:
    """
    Preprocessor for text, audio, and video features in the dataset.

    Methods
    -------
    load_features() -> Tuple[List, List, List, List, List, List, List, List, List]:
        Load audio, audio-IS, and video features from pickle files.
    tokenize_text(examples: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Tokenize text data into input IDs, attention masks, and token type IDs.
    preprocess_data(text_data: List[str], audio_data, video_data, audio_is_data, labels) -> TensorDataset:
        Preprocess text, audio, video, and label data into a TensorDataset.
    prepare_dataloader(data: TensorDataset, batch_size: int, sampler_type: str = "random") -> DataLoader:
        Create a DataLoader for the given dataset.
    """
    def __init__(self, config: Config):
        """
        Initialize the preprocessor with configuration.

        Parameters
        ----------
        config : Config
            Configuration object containing dataset paths and parameters.
        """
        self.config = config

    def load_tsv(self, file_path: str) -> Tuple[List[str], List[float]]:
        """
        Load text and labels from a TSV file.

        Parameters
        ----------
        file_path : str
            The path to the TSV file.

        Returns
        -------
        Tuple[List[str], List[float]]
            A tuple containing a list of text data and corresponding labels.
        """
        texts, labels = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                texts.append(row[0])  # 假设第一列是文本
                labels.append(float(row[1]))  # 假设第二列是标签
        return texts, labels

    def load_features(self) -> Tuple[List, List, List, List, List, List, List, List, List]:
        """
        Load audio, audio-IS, and video features.

        Returns
        -------
        Tuple[List, List, List, List, List, List, List, List, List]
            Train, validation, and test sets for audio, audio-IS, and video features.
        """
        with open(self.config.audio_file, "rb") as f:
            train_audio, valid_audio, test_audio = pickle.load(f)

        with open(self.config.audio_is_file, "rb") as f:
            train_audio_IS, valid_audio_IS, test_audio_IS = pickle.load(f)

        with open(self.config.video_file, "rb") as f:
            train_video, valid_video, test_video = pickle.load(f)

        # 处理 audio-IS 数据
        train_audio_IS = self._process_audio_is(train_audio_IS)
        valid_audio_IS = self._process_audio_is(valid_audio_IS)
        test_audio_IS = self._process_audio_is(test_audio_IS)

        return (train_audio, valid_audio, test_audio,
                train_audio_IS, valid_audio_IS, test_audio_IS,
                train_video, valid_video, test_video)

    def _process_audio_is(self, audio_is_data: list) -> np.ndarray:
        """
        Process audio-IS data to ensure it is a valid numpy array.

        Parameters
        ----------
        audio_is_data : list
            Raw audio-IS data.

        Returns
        -------
        np.ndarray
            Processed audio-IS data as a numpy array.
        """
        # 检查嵌套结构并处理
        try:
            return np.array(audio_is_data, dtype=np.float32)
        except ValueError:
            max_len = max(len(item) if isinstance(item, list) else 0 for item in audio_is_data)
            return np.array([
                np.pad(item, (0, max_len - len(item)), constant_values=0).astype(np.float32)
                if isinstance(item, list) else np.zeros(max_len, dtype=np.float32)
                for item in audio_is_data
            ])

    def tokenize_text(self, examples: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize and encode text data.

        Parameters
        ----------
        examples : List[str]
            A list of input sentences to be tokenized.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Encoded input IDs, attention masks, and token type IDs.
        """
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

    def preprocess_data(self, text_data: List[str], audio_data, video_data, audio_is_data, labels) -> TensorDataset:
        """
        Preprocess text, audio, video, and label data into TensorDataset.

        Parameters
        ----------
        text_data : List[str]
            A list of text data.
        audio_data : list
            List of audio features.
        video_data : list
            List of video features.
        audio_is_data : list
            List of audio-IS features.
        labels : list
            List of labels.

        Returns
        -------
        TensorDataset
            A dataset containing tokenized text, audio, video, and labels.
        """
        # 确保 audio_is_data 是数值类型
        audio_is_data = torch.tensor(audio_is_data, dtype=torch.float32).unsqueeze(1)

        input_ids, attention_masks, token_type_ids = self.tokenize_text(text_data)
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
        video_data = torch.tensor(video_data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return TensorDataset(input_ids, attention_masks, token_type_ids, audio_data, labels, audio_is_data, video_data)

    def prepare_dataloader(self, data: TensorDataset, batch_size: int, sampler_type: str = "random") -> DataLoader:
        """
        Create a DataLoader for the given dataset.

        Parameters
        ----------
        data : TensorDataset
            The dataset to be loaded.
        batch_size : int
            The batch size for loading data.
        sampler_type : str, optional
            The sampler type ("random" or "sequential"), by default "random".

        Returns
        -------
        DataLoader
            The DataLoader for the dataset.
        """
        if sampler_type == "random":
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=batch_size)


def prepare_mosi_datasets(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    processor = DataPreprocessor(config)

    # 加载音频、视频特征
    (train_audio, valid_audio, test_audio,
     train_audio_IS, valid_audio_IS, test_audio_IS,
     train_video, valid_video, test_video) = processor.load_features()

    # 从 TSV 文件加载文本和标签
    train_texts, train_labels = processor.load_tsv(os.path.join(config.text_dir, "ptrain.tsv"))
    valid_texts, valid_labels = processor.load_tsv(os.path.join(config.text_dir, "pvalid.tsv"))
    test_texts, test_labels = processor.load_tsv(os.path.join(config.text_dir, "ptest.tsv"))

    # 数据预处理
    train_data = processor.preprocess_data(train_texts, train_audio, train_video, train_audio_IS, train_labels)
    valid_data = processor.preprocess_data(valid_texts, valid_audio, valid_video, valid_audio_IS, valid_labels)
    test_data = processor.preprocess_data(test_texts, test_audio, test_video, test_audio_IS, test_labels)

    # 创建 DataLoaders
    train_loader = processor.prepare_dataloader(train_data, config.train_batch_size, sampler_type="random")
    valid_loader = processor.prepare_dataloader(valid_data, config.train_batch_size, sampler_type="sequential")
    test_loader = processor.prepare_dataloader(test_data, config.train_batch_size, sampler_type="sequential")
    return train_loader, valid_loader, test_loader


def load_and_inspect_labels(file_path):
    """
    Load and display labels and decoded text from a TSV file.

    Parameters
    ----------
    file_path : str
        The path to the TSV file.

    Returns
    -------
    None
    """
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            text, label = row[0], float(row[1])  # 假设第一列是文本，第二列是标签
            print(f"Row {i + 1} -> Text: {text}, Label: {label}")


if __name__ == "__main__":
    # 创建配置对象，设置数据库路径
    config = Config(dataset="D:/Project/Modility_Fusion_Seqence/Data/MOSI")

    # 准备数据集
    train_loader, valid_loader, test_loader = prepare_mosi_datasets(config)

    # 打印数据加载结果
    for batch in train_loader:
        input_ids, attention_mask, token_type_ids, audio, labels, audio_is, video = batch
        print("Batch shapes:", input_ids.shape, audio.shape, video.shape)

