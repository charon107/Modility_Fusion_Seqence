import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, AutoModel
from transformers import BertTokenizer
from transformers import ViTModel
from torch.utils.data import DataLoader
from typing import List
import numpy as np
from dataloader import Config


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, nhead=1, dim_feedforward=128, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, attention_mask=None):
        src_1 = self.self_attention(src, src, src_key_padding_mask=attention_mask)
        src = src + self.dropout1(src_1)
        src = self.norm1(src)
        src_2 = self.self_attention(src, src, src_key_padding_mask=attention_mask)
        src = src + self.dropout2(src_2)
        src = self.norm2(src)
        return src



class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, src, attention_mask=None):
        return self.transformer_encoder(src, src_key_padding_mask=attention_mask)


class Transformer(nn.Module):
    def __init__(self, d_model, num_layers=1, nhead=1, dropout=0.1, dim_feedforward=128, max_seq_length=5000,
                 input_dim=1582):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 256)
        self.norm = nn.LayerNorm(d_model)
        self.input_projection = nn.Linear(input_dim, d_model)

    def forward(self, input1):
        """
        输入:
        - input1: 输入张量，形状为 [batch_size, seq_length, input_dim]
        - attention_mask: 可选的注意力掩码，用于Transformer中的注意力机制

        返回:
        - out: 解码后的结果
        - hidden: Transformer encoder的输出
        """

        seq_length = input1.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input1.device)
        positions_embedding = self.pos_encoder(position_ids).unsqueeze(0).expand(input1.size(0), -1, -1)
        input1 = self.input_projection(input1)
        input1 = input1 + positions_embedding
        input1 = self.norm(input1)
        hidden = self.encoder(input1)
        out = self.decoder(hidden)

        return out, hidden


class ModalityEncoder(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased", video_model_name=None,
                 audio_d_model=256, audio_layers=2, audio_heads=4, audio_dropout=0.1):
        super(ModalityEncoder, self).__init__()
        # Text Encoder (BERT)
        self.text_encoder = BertModel.from_pretrained(text_model_name)


        # Audio Encoder (using Transformer)
        self.audio_encoder = Transformer(d_model=audio_d_model, num_layers=audio_layers, nhead=audio_heads,
                                         dropout=audio_dropout)

        # Video Encoder (ViT or custom)
        self.video_encoder = ViTModel.from_pretrained(
            video_model_name) if video_model_name else AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.text_feature_projection = nn.Linear(768,256)
        self.video_feature_projection = nn.Linear(711,256)

    def forward(self, text_inputs, text_masks, audio_inputs, video_inputs):
        # Text encoding using BERT
        text_outputs = self.text_encoder(input_ids=text_inputs, attention_mask=text_masks).last_hidden_state
        text_outputs = self.text_feature_projection(text_outputs)

        # Audio encoding
        audio_outputs, _ = self.audio_encoder(audio_inputs)

        # Video encoding (assuming the last_hidden_state exists for ViT models)
        if hasattr(self.video_encoder, 'last_hidden_state'):
            video_outputs = self.video_encoder(video_inputs).last_hidden_state
        else:
            video_outputs = video_inputs
        video_outputs = self.video_feature_projection(video_outputs)


        return text_outputs, audio_outputs, video_outputs

class CrossModalTransformer(nn.Module):
    """
    Implements Cross-Modal Transformer (CMT) logic using the MyCSA1 and my_attention1 structure.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1, layers=2):
        super(CrossModalTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [MyCSA1(embed_dim, embed_dim, nhead=num_heads, dropout=dropout) for _ in range(layers)]
        )

    def forward(self, base, addition):
        for layer in self.layers:
            addition = layer(base, addition)
        return addition


class MyCSA1(nn.Module):
    """
    Cross-Modal Transformer layer for modality fusion.
    """

    def __init__(self, text_size, addition_size, nhead, dropout):
        super(MyCSA1, self).__init__()
        self.cross_attention = my_attention1(text_size, addition_size, nhead, dropout)

    def forward(self, base, addition):
        return self.cross_attention(base, addition)


class my_attention1(nn.Module):
    """
    Attention mechanism using textual features as the base and other modalities as addition.
    """

    def __init__(self, hidden_size, context_size, nhead, dropout):
        super(my_attention1, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=context_size, num_heads=nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(context_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(context_size, context_size), nn.ReLU(), nn.Linear(context_size, context_size))

    def forward(self, base, addition):
        addition, _ = self.cross_attention(query=addition, key=base, value=base)
        addition = self.norm1(addition + self.dropout1(self.fc(addition)))
        return addition


class RobertaProcessor(nn.Module):
    """
    Roberta-based processing for multimodal fused features.

    Parameters
    ----------
    roberta_model_name : str
        The name of the pre-trained RoBERTa model (default: 'roberta-base').
    embed_dim : int
        The embedding dimension of the input features.
    """

    def __init__(self, roberta_model_name="roberta-base", embed_dim=256):
        super(RobertaProcessor, self).__init__()
        self.roberta_model = RobertaModel.from_pretrained(roberta_model_name)
        self.feature_projection = nn.Linear(embed_dim, self.roberta_model.config.hidden_size)
        self.output_projection = nn.Linear(self.roberta_model.config.hidden_size, embed_dim)

    def forward(self, text_audio_video_features):
        """
        Forward pass through the RobertaProcessor.

        Parameters
        ----------
        text_audio_video_features : torch.Tensor
            The input tensor of shape [batch_size, seq_length, embed_dim].

        Returns
        -------
        torch.Tensor
            Processed features of shape [batch_size, seq_length, embed_dim].
        """

        projected_features = self.feature_projection(text_audio_video_features)
        batch_size, seq_length, _ = projected_features.size()
        flattened_features = projected_features.view(batch_size, seq_length, -1)
        roberta_outputs = self.roberta_model(inputs_embeds=flattened_features).last_hidden_state
        reshaped_outputs = roberta_outputs.view(batch_size, seq_length, -1)
        processed_features = self.output_projection(reshaped_outputs)

        return processed_features


class MultimodalFusion(nn.Module):
    """
    Multimodal Fusion using Cross-Modal Transformers (CMT).
    First fuses text and audio, then fuses text+audio with video.

    Parameters
    ----------
    embed_dim : int
        The embedding dimension for the feature projection and transformer layers.
    num_heads : int
        The number of attention heads for the Cross-Modal Transformers.
    num_layers : int
        The number of layers for the Cross-Modal Transformers.
    dropout : float, optional, default=0.1
        The dropout rate for regularization.
    output_dim : int, optional, default=1
        The output dimension for the final regression layer.

    Attributes
    ----------
    modality_encoder : nn.Module
        The modality encoder that processes text, audio, and video inputs.
    text_audio_cmt : nn.ModuleList
        The list of Cross-Modal Transformer layers for text and audio fusion.
    text_audio_video_cmt : nn.ModuleList
        The list of Cross-Modal Transformer layers for text, audio, and video fusion.
    roberta_processor : nn.Module
        The RobertaProcessor that processes the fused features.
    """

    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1, output_dim=1, roberta_model_name="roberta-base"):
        super(MultimodalFusion, self).__init__()
        self.modality_encoder = ModalityEncoder()  # Define ModalityEncoder here
        self.text_audio_cmt = CrossModalTransformer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                                                    layers=num_layers)
        self.text_audio_video_cmt = CrossModalTransformer(embed_dim=embed_dim, num_heads=4, dropout=dropout,
                                                          layers=num_layers)
        self.roberta_processor = RobertaProcessor(roberta_model_name=roberta_model_name, embed_dim=embed_dim)

        self.linear_layer = nn.Linear(embed_dim, 2)
        self.sigmoid_variant = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Tanh()
        )




    def forward(self, text_inputs, text_masks, audio_inputs, video_inputs):
        """
        Forward pass through the model, returning the output after feature fusion, pooling, and regression.

        Parameters
        ----------
        text_inputs : torch.Tensor
            The input tensor for text data.
        text_masks : torch.Tensor
            The attention masks for text inputs.
        audio_inputs : torch.Tensor
            The input tensor for audio data.
        video_inputs : torch.Tensor
            The input tensor for video data.

        Returns
        -------
        torch.Tensor
            Processed features of shape [batch_size, seq_length, embed_dim].
        """
        text_outputs, audio_outputs, video_outputs = self.modality_encoder(text_inputs, text_masks, audio_inputs, video_inputs)
        text_audio_features = self.text_audio_cmt(text_outputs, audio_outputs)
        video_features = video_outputs
        text_audio_video_features = self.text_audio_video_cmt(text_audio_features, video_features)
        roberta_features = self.roberta_processor(text_audio_video_features)
        roberta_features = self.linear_layer(roberta_features[:, 0, :])
        prediction = self.sigmoid_variant(roberta_features)

        return prediction




