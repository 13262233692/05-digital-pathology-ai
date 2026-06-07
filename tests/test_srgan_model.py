import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.srgan import Generator, Discriminator, SRGAN


def test_generator_forward():
    generator = Generator(
        in_channels=3,
        num_filters=64,
        num_residual_blocks=16,
        scale_factor=4
    )
    
    batch_size = 2
    input_size = 128
    x = torch.randn(batch_size, 3, input_size, input_size)
    
    with torch.no_grad():
        output = generator(x)
    
    expected_size = input_size * 4
    assert output.shape == (batch_size, 3, expected_size, expected_size)
    
    print("✓ Generator forward pass test passed")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")


def test_discriminator_forward():
    discriminator = Discriminator(in_channels=3, num_filters=64)
    
    batch_size = 2
    input_size = 512
    x = torch.randn(batch_size, 3, input_size, input_size)
    
    with torch.no_grad():
        output = discriminator(x)
    
    assert output.shape == (batch_size, 1)
    assert (output >= 0).all() and (output <= 1).all()
    
    print("✓ Discriminator forward pass test passed")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")


def test_srgan_build():
    device = torch.device("cpu")
    srgan = SRGAN.build(
        in_channels=3,
        num_filters=64,
        num_residual_blocks=8,
        scale_factor=4,
        device=device
    )
    
    x = torch.randn(1, 3, 128, 128)
    
    with torch.no_grad():
        sr_output = srgan(x)
    
    assert sr_output.shape == (1, 3, 512, 512)
    
    print("✓ SRGAN build test passed")


def test_512_tile_inference():
    generator = Generator(
        in_channels=3,
        num_filters=64,
        num_residual_blocks=16,
        scale_factor=4
    )
    generator.eval()
    
    tile_size = 512
    x = torch.randn(1, 3, tile_size, tile_size)
    
    with torch.no_grad():
        output = generator(x)
    
    assert output.shape == (1, 3, 2048, 2048)
    
    print("✓ 512x512 tile inference test passed")
    print(f"  Input: {tile_size}x{tile_size} → Output: 2048x2048 (4x SR)")


if __name__ == "__main__":
    test_generator_forward()
    test_discriminator_forward()
    test_srgan_build()
    test_512_tile_inference()
    print("\nAll tests passed!")
