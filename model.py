import torch
import torch.nn as nn
import random
import numpy as np
from transformers import AutoModel
import torch.nn.functional as F


def set_random_seed(seed):
    """
    Set the random seed for reproducibility of experiments.

    Parameters
    ----------
    seed : int
        The seed value to set for randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility
    torch.backends.cudnn.benchmark = False  # Disable CuDNN auto-tuning for deterministic results

# Set a fixed random seed
seed = 45
set_random_seed(seed)

class AudioFeatureEncoder(nn.Module):
    """
    Transformer-based audio feature encoder with positional embeddings.
    """

    def __init__(self,
                 input_dim=1582,
                 embed_dim=256,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 max_seq_length=5000):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Positional embeddings
        self.position_emb = nn.Embedding(max_seq_length, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass through the audio feature encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_len, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, embed_dim)
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # Input projection
        x = self.input_proj(x)  # [B, T, D]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_emb(positions).unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb

        # Encode sequence
        encoded = self.encoder(x)

        # Aggregate features
        return self.layer_norm(encoded.mean(1))

class VideoFeatureEncoder(nn.Module):
    """
    Transformer-based encoder for processing 711-dimensional OpenFace video features.
    """

    def __init__(self, input_dim=711, d_model=512, num_layers=4, nhead=8, dropout=0.1, dim_feedforward=1024, max_seq_length=5000):
        super().__init__()

        self.d_model = d_model

        # Linear projection: project 711-dimensional features to Transformer hidden size d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        nn.init.xavier_normal_(self.input_proj.weight)

        # Positional encoding (Embedding)
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.pre_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model // 2)
        self.norm = nn.LayerNorm(d_model // 2)

    def check_nan(self, x, layer_name):
        """
        Check if tensor x contains NaN values and print a warning.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to check for NaN values.
        layer_name : str
            The name of the layer for logging.

        Raises
        ------
        ValueError
            If NaN values are detected in the tensor.
        """
        if torch.isnan(x).any():
            print(f"NaN detected at {layer_name}:")
            print(f" - Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f" - Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
            raise ValueError("NaN detected")

    def forward(self, x):
        """
        Forward pass through the video feature encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_len, 711)

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, d_model // 2)
        """

        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

        # Linear projection to d_model dimension
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        x = self.pre_norm(x)
        if self.check_nan(x, 'input_proj'):
            raise ValueError("NaN detected after input_proj")

        batch_size, seq_len , _ = x.shape

        # Add positional encoding
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_embeddings = self.pos_encoder(position_ids).unsqueeze(0).expand(batch_size, -1, -1)
        x = x + position_embeddings  # Add temporal information
        if self.check_nan(x, 'position_embeddings'):
            raise ValueError("NaN detected after adding position_embeddings")

        # Transformer encoding
        x = self.encoder(x)
        if self.check_nan(x, 'Transformer Encoder'):
            raise ValueError("NaN detected after transformer encoder")

        # Pooling: Take [CLS] token or global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        if self.check_nan(x, 'mean pooling'):
            raise ValueError("NaN detected after mean pooling")

        # Projection and normalization
        x = self.output_proj(x)
        if self.check_nan(x, 'output_proj'):
            raise ValueError("NaN detected after output_proj")

        x = self.norm(x)
        if self.check_nan(x, 'LayerNorm'):
            raise ValueError("NaN detected after LayerNorm")

        return x

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer with residual connections.
    """

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.constant_(self.attention.in_proj_bias, 0.)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        nn.init.constant_(self.attention.out_proj.bias, 0.)

    def forward(self, query, context):
        """
        Perform cross-attention between the query and context.

        Parameters
        ----------
        query : torch.Tensor
            The main modality features [B, D]
        context : torch.Tensor
            The contextual modality features [B, D]

        Returns
        -------
        torch.Tensor
            The output tensor after attention and feed-forward processing.
        """
        query = query.unsqueeze(1)  # [B, 1, D]
        context = context.unsqueeze(1)  # [B, 1, D]

        # Cross-attention
        attended, _ = self.attention(
            query=query,
            key=context,
            value=context
        )

        # Residual connection
        attended = query + self.dropout(attended)
        attended = self.norm(attended)
        attended = self.norm(query + self.dropout(attended))

        # Feed-forward
        output = attended + 0.5 * self.dropout(self.ffn(attended))
        return output.squeeze(1)

class MultimodalFusionNetwork(nn.Module):
    """
    Multimodal fusion network with hierarchical cross-modal attention and Roberta processing.
    """

    def __init__(self,
                 text_model="bert-base-uncased",
                 audio_feature_dim=1582,
                 video_feature_dim=711,
                 embed_dim=256,
                 num_heads=8,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()

        # Modality encoders
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.text_proj = nn.Linear(768, embed_dim)
        self.audio_norm = nn.LayerNorm(audio_feature_dim)
        self.video_norm = nn.LayerNorm(video_feature_dim)

        self.audio_encoder = AudioFeatureEncoder(
            input_dim=audio_feature_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        self.video_encoder = VideoFeatureEncoder(
            input_dim=video_feature_dim,
            d_model=embed_dim * 2,
            num_layers=num_layers,
            nhead=num_heads,
            dropout=dropout
        )

        # Cross-modal attention layers
        self.text_audio_attn = CrossModalAttention(embed_dim, num_heads, dropout)
        self.text_audio_video_attn = CrossModalAttention(embed_dim, num_heads, dropout)

        # Roberta processor
        self.roberta_processor = AutoModel.from_pretrained("roberta-base")
        self.roberta_proj = nn.Linear(embed_dim, self.roberta_processor.config.hidden_size)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta_processor.config.hidden_size, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 1),
            nn.Tanh()
        )

    def forward(self, text_inputs, text_masks, audio_inputs, video_inputs):
        """
        Forward pass with Roberta processing.

        Parameters
        ----------
        text_inputs : torch.Tensor
            Input text token IDs.
        text_masks : torch.Tensor
            Text attention masks.
        audio_inputs : torch.Tensor
            Audio features.
        video_inputs : torch.Tensor
            Video features.

        Returns
        -------
        torch.Tensor
            Final output predictions.
        """
        with torch.autograd.set_detect_anomaly(True):
            audio_inputs = self.audio_norm(audio_inputs)
            video_inputs = self.video_norm(video_inputs)

            text_features = self.text_encoder(
                input_ids=text_inputs,
                attention_mask=text_masks
            ).last_hidden_state.mean(1)
            text_features = self.text_proj(text_features)

            audio_features = self.audio_encoder(audio_inputs)
            video_features = self.video_encoder(video_inputs)

            text_audio = self.text_audio_attn(text_features, audio_features)
            text_audio_video_features = self.text_audio_video_attn(text_audio, video_features)

            roberta_input = self.roberta_proj(text_audio_video_features).unsqueeze(1)
            roberta_output = self.roberta_processor(inputs_embeds=roberta_input).last_hidden_state
            roberta_features = roberta_output.mean(1)

            output = self.classifier(roberta_features)
            return output