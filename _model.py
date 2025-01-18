import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, RobertaModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from timm.models.vision_transformer import vit_base_patch16_224


class ModalityEncoder(nn.Module):
    """
    Encodes inputs from different modalities using pre-trained models.
    """

    def __init__(self, text_model_name="bert-base-uncased", audio_model_name="facebook/wav2vec2-base-960h",
                 video_model_name=None):
        super(ModalityEncoder, self).__init__()
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.video_encoder = vit_base_patch16_224(pretrained=True) if video_model_name is None else AutoModel.from_pretrained(video_model_name)

    def forward(self, text_inputs, text_masks, audio_inputs, video_inputs):
        # Text encoding
        text_outputs = self.text_encoder(input_ids=text_inputs, attention_mask=text_masks).last_hidden_state

        # Audio encoding
        audio_outputs = self.audio_encoder(audio_inputs).last_hidden_state  # Wav2Vec2 output

        # Video encoding
        video_outputs = self.video_encoder(video_inputs).last_hidden_state if hasattr(self.video_encoder,
                                                                                      'last_hidden_state') else video_inputs

        return text_outputs, audio_outputs, video_outputs


class CrossModalTransformer(nn.Module):
    """
    Implements Cross-Modal Transformer (CMT) logic using the MyCSA1 and my_attention1 structure.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1, layers=2):
        super(CrossModalTransformer, self).__init__()
        self.layers = nn.ModuleList([
            MyCSA1(embed_dim, embed_dim, nhead=num_heads, dropout=dropout) for _ in range(layers)
        ])

    def forward(self, base, addition):
        for layer in self.layers:
            addition = layer(base, addition)  # MyCSA1 implements the logic for CMT
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
        # Base is used as key and value, addition as query
        addition, _ = self.cross_attention(query=addition, key=base, value=base)
        addition = self.norm1(addition + self.dropout1(self.fc(addition)))
        return addition


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion model with CMT and final regression.
    """

    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1, output_dim=1):
        super(MultimodalFusion, self).__init__()
        self.text_audio_cmt = CrossModalTransformer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                                                    layers=num_layers)
        self.text_audio_video_cmt = CrossModalTransformer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                                                          layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.regressor = nn.Linear(embed_dim, output_dim)

    def forward(self, text_features, audio_features, video_features):
        # Stage 1: Text + Audio fusion
        text_audio_features = self.text_audio_cmt(text_features, audio_features)

        # Stage 2: Text-Audio + Video fusion
        text_audio_video_features = self.text_audio_video_cmt(text_audio_features, video_features)

        # Pooling and regression
        pooled_features = self.pooling(text_audio_video_features.permute(0, 2, 1)).squeeze(-1)
        output = self.regressor(self.dropout(pooled_features))
        return output


class MultimodalLearningModel(nn.Module):
    """
    Overall model combining modality encoders, CMT, and final prediction layer.
    """

    def __init__(self, text_model_name="bert-base-uncased", audio_model_name="facebook/wav2vec2-base-960h",
                 video_model_name=None, embed_dim=768, num_heads=8, num_layers=4, dropout=0.1):
        super(MultimodalLearningModel, self).__init__()
        self.modality_encoder = ModalityEncoder(text_model_name, audio_model_name, video_model_name)
        self.fusion_model = MultimodalFusion(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                                             dropout=dropout)

    def forward(self, text_inputs, text_masks, audio_inputs, video_inputs):
        text_features, audio_features, video_features = self.modality_encoder(text_inputs, text_masks, audio_inputs,
                                                                              video_inputs)
        output = self.fusion_model(text_features, audio_features, video_features)
        return output


# Example usage
if __name__ == "__main__":
    # Input dimensions
    batch_size = 32
    seq_length = 50
    audio_dim = 768
    video_dim = 768
    embed_dim = 768

    # Mock inputs
    text_inputs = torch.randint(0, 30522, (batch_size, seq_length))  # Random token IDs for BERT
    text_masks = torch.ones(batch_size, seq_length)  # Attention masks
    audio_inputs = torch.randn(batch_size, seq_length, audio_dim)
    video_inputs = torch.randn(batch_size, seq_length, video_dim)

    # Model initialization
    model = MultimodalLearningModel(embed_dim=embed_dim, num_heads=8, num_layers=4)
    outputs = model(text_inputs, text_masks, audio_inputs, video_inputs)

    print("Output shape:", outputs.shape)
