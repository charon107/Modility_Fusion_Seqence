import torch
import torch.nn as nn
from transformers import AutoModel


class AudioFeatureEncoder(nn.Module):
    """Transformer-based audio feature encoder with positional embeddings"""

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
        Input: (batch_size, seq_len, input_dim)
        Output: (batch_size, embed_dim)
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
    """Hierarchical video feature encoder with temporal modeling"""

    def __init__(self,
                 input_dim=711,
                 hidden_dim=512,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()

        # Feature enhancement
        self.feature_enhancer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Temporal encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, hidden_dim))

        # Feature compression
        self.compressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )

    def forward(self, x):
        """
        Input: (batch_size, seq_len, input_dim)
        Output: (batch_size, hidden_dim//2)
        """
        # Feature enhancement
        x = self.feature_enhancer(x)  # [B, T, H]

        # Add position embeddings
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        # Temporal encoding
        x = self.temporal_encoder(x)

        # Feature compression
        return self.compressor(x.mean(1))


class CrossModalAttention(nn.Module):
    """Cross-modal attention layer with residual connections"""

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

    def forward(self, query, context):
        """
        query: main modality features [B, D]
        context: contextual modality features [B, D]
        """
        # Reshape for attention
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

        # Feed-forward
        output = attended + self.dropout(self.ffn(attended))
        return output.squeeze(1)


class MultimodalFusionNetwork(nn.Module):
    """Multimodal fusion network with hierarchical cross-modal attention"""

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

        self.audio_encoder = AudioFeatureEncoder(
            input_dim=audio_feature_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        self.video_encoder = VideoFeatureEncoder(
            input_dim=video_feature_dim,
            hidden_dim=embed_dim * 2,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # Cross-modal attention layers
        self.text_audio_attn = CrossModalAttention(embed_dim, num_heads, dropout)
        self.text_video_attn = CrossModalAttention(embed_dim, num_heads, dropout)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_inputs, text_masks, audio_inputs, video_inputs):
        # Text processing
        text_features = self.text_encoder(
            input_ids=text_inputs,
            attention_mask=text_masks
        ).last_hidden_state.mean(1)
        text_features = self.text_proj(text_features)  # [B, D]

        # Audio processing
        audio_features = self.audio_encoder(audio_inputs)  # [B, D]

        # Video processing
        video_features = self.video_encoder(video_inputs)  # [B, D]

        # Cross-modal attention
        text_audio = self.text_audio_attn(text_features, audio_features)
        text_video = self.text_video_attn(text_features, video_features)

        # Multimodal fusion
        fused = torch.cat([text_audio, text_video], dim=1)  # [B, 2D]

        return self.classifier(fused)