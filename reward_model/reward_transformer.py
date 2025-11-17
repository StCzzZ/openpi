import torch
import torch.nn as nn



class RewardTransformer(nn.Module):
    def __init__(self, args, video_dim=768, text_dim=384, hidden_dim=512, num_heads=8, num_layers=4, num_stages=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.args = args
        
        # Project video and text to common dimension
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Position embeddings for video sequence
        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))  # 32 is max_length
        
        # Class token embedding
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Shared progress prediction head (applied to each frame)
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Stage classification head (applied to each frame)
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_stages),
            nn.Softmax(dim=-1)
        )

        # Progress estimation head (applied to each frame)
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Attention mask for causal self-attention
        self.attention_mask = nn.Transformer.generate_square_subsequent_mask(args.max_seq_len + 1).to('cuda')

    def forward(self, video_frames, text_embed):
        
        # Project inputs to common dimension
        video_embed = self.video_proj(video_frames)  # [batch_size, seq_len, hidden_dim]
        text_embed = self.text_proj(text_embed)  # [batch_size, 1, hidden_dim]
        
        # Add positional embeddings to video]
        video_embed[:,0] += self.first_pos_embed
                
        # Combine sequence: [text, video_frames]
        sequence = torch.cat([text_embed, video_embed], dim=1)
        
        # Pass through transformer
        transformed = self.transformer(sequence, is_causal=True, mask = self.attention_mask)
        
        # Get progress predictions for each frame
        stage_embedding = self.shared_head(transformed[:, 1:])  # Exclude class token and text token

        stage_preds = self.stage_head(stage_embedding)
        progress_preds = self.progress_head(stage_embedding)
        
        return stage_preds, progress_preds