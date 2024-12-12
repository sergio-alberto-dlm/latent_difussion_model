import numpy as np 
import torch 
import torch.nn as nn

def get_time_embedding(time_steps, t_emb_dim):
    "positional encoding for time-steps"
    factor = 10000 ** ((torch.arange(
        start=0, end=t_emb_dim//2, device=time_steps.device) / (t_emb_dim // 2)
    ))
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor 
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

def positional_encoding(sequence_lenth, channels):
    "positional encoding por visual patches"
    def embeding(pos):
        emb = np.zeros(channels)
        for i in range(channels):
            if i % 2 == 0:
                emb[i] = np.sin(pos / (10 ** (i / sequence_lenth)))
            else:
                emb[i] = np.cos(pos / (10 ** ((i - 1) / sequence_lenth)))
        return emb

    encoding = [torch.tensor(embeding(pos), dtype=torch.float32) for pos in range(sequence_lenth)]

    return torch.stack(encoding)

# transformer architecture 
class TransformerBlock(nn.Module):
    "transformer block with time-step information merge"
    
    def __init__(self, channels, t_emb_dim, num_heads) -> tuple:
        super().__init__()
        self.mhattn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.lnorm1 = nn.LayerNorm(channels)
        
        self.mlp_sub_block = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(t_emb_dim, channels)
        )
        self.attn_mat = None

    def forward(self, x, t_emb):
        id      = x.clone()
        # multihead attn sub-block 
        x = x + self.t_emb_layers(t_emb) # include time-step information
        output  = self.lnorm1(x)                                           
        output, attn_mat  = self.mhattn(output, output, output)   
        # print("prev output: ", output.shape)
        # print("t_emb_layers: ", self.t_emb_layers(t_emb).shape)                                                
        output  = output + id                                                  
        id      = output.clone()

        # mlp sub-block
        output = self.mlp_sub_block(output)
        output += id   
        
        self.attn_mat = attn_mat                                                    
                                                        
        return output

class Transformer(nn.Module):
    "transformer model"
    def __init__(self, channels, t_emb_dim, num_heads, patch_size, sequence_length, num_blocks) -> torch.tensor:
        super().__init__()
        self.channels = channels
        self.t_emb_dim = t_emb_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.num_blocks = num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(channels, t_emb_dim, num_heads) for _ in range(num_blocks)]
        )
        self.net = nn.Sequential(*self.transformer_blocks)

        self.linear_proj = nn.Linear(patch_size * patch_size * 3, channels)

        self.positional_encodings = positional_encoding(sequence_length, channels)

    def get_image_patches(self, batch_images, patch_size):
        B, C, H, W = batch_images.shape
        assert H % patch_size == 0 and W % patch_size == 0, "Height and Width must be divisible by patch size"

        # Extract patches
        patches = batch_images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # Shape: (B, num_patches_h, num_patches_w, C, patch_size, patch_size)

        # Reshape to merge all patches per image
        num_patches_h, num_patches_w = patches.shape[1:3]
        patches = patches.reshape(B, num_patches_h * num_patches_w, C, patch_size, patch_size)  # Shape: (B, num_patches, C, patch_size, patch_size)
        # Flatten patches
        flattened_patches = patches.view(B, patches.shape[1], -1)  # Shape: (B, num_patches, C * patch_size^2)

        return flattened_patches

    def forward(self, images, t_emb):
        x = self.get_image_patches(batch_images=images, patch_size=self.patch_size)
        x = self.linear_proj(x)
        x = x + self.positional_encodings
        for block in self.transformer_blocks:
            x = block(x, t_emb)
        return x