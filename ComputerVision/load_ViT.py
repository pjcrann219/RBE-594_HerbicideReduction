import torch
from transformers import ViTForImageClassification, ViTConfig
from torchvision import transforms

def get_ViT(pretrained_model_name="google/vit-base-patch16-224"):
    """
    Load a pretrained ViT model and modify it for custom input channels and finetuning.
    """

    num_channels=4
    num_classes=1

    # Load the pretrained model configuration
    config = ViTConfig.from_pretrained(pretrained_model_name)
    
    # Update the configuration for custom number of channels
    config.num_channels = num_channels
    config.num_labels = num_classes
    config.image_size = 512
    
    # Initialize model with modified config
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name,
        config=config,
        ignore_mismatched_sizes=True  # Needed when changing input channels
    )
    
    # Modify the embedding layer to handle custom number of channels
    old_embeddings = model.vit.embeddings.patch_embeddings.projection
    new_embeddings = torch.nn.Conv2d(
        in_channels=num_channels,
        out_channels=old_embeddings.out_channels,
        kernel_size=old_embeddings.kernel_size,
        stride=old_embeddings.stride,
        padding=old_embeddings.padding
    )
    
    # Average the pretrained weights across channels and replicate for new channels
    with torch.no_grad():
        new_embeddings.weight[:, :3, :, :] = old_embeddings.weight[:, :3, :, :]
        new_embeddings.weight[:, 3:, :, :] = old_embeddings.weight[:, :3, :, :].mean(dim=1, keepdim=True)
    
    # Replace the embedding layer
    model.vit.embeddings.patch_embeddings.projection = new_embeddings
    
    pos_embed = model.vit.embeddings.position_embeddings
    num_patches = (512 // 16) ** 2  # 32x32 patches for 512x512 image
    pos_embed_resized = interpolate_pos_encoding(pos_embed, num_patches)
    model.vit.embeddings.position_embeddings.data = pos_embed_resized

    # Freeze backbone if specified
    for param in model.vit.encoder.parameters():
        param.requires_grad = False

    
    return model

def interpolate_pos_encoding(pos_embed, num_patches):
    """Helper function to interpolate position embeddings"""
    N = pos_embed.shape[1] - 1  # number of patches in original embedding
    if N == num_patches:
        return pos_embed
        
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    
    dim = pos_embed.shape[-1]
    h_old = w_old = int(N ** 0.5)
    h_new = w_new = int(num_patches ** 0.5)
    
    patch_pos_embed = patch_pos_embed.reshape(1, h_old, w_old, dim)
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed.permute(0, 3, 1, 2),
        size=(h_new, w_new),
        mode='bicubic',
        align_corners=False
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
    
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

# Example usage:
if __name__ == "__main__":
    # Load model with 4 input channels and 10 output classes
    model = get_ViT()
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 4, 512, 512)
    
    # Test forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.logits.shape}")
