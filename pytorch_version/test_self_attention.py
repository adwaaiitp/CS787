import torch
import sys
sys.path.append('training')
from networks import SelfAttention, SelfAttentionWrapper, SelfAttentionNoLatent

print("Testing Self-Attention implementations...")

# Test 1: Pure Self-Attention
print("\n1. Testing SelfAttention (X-X):")
self_attn = SelfAttention(channels=512, num_heads=8)
f = torch.randn(2, 16, 16, 512)  # (B, H, W, C)
f_out = self_attn(f)
print(f"   Input: {f.shape} -> Output: {f_out.shape}")
assert f.shape == f_out.shape, "Shape mismatch!"
print("   ✓ Passed")

# Test 2: Wrapper (compatible with BipartiteAttention interface)
print("\n2. Testing SelfAttentionWrapper (compatible interface):")
wrapper = SelfAttentionWrapper(channels=512, components_num=8, latent_size=128, num_heads=8)
z = torch.randn(2, 8, 128)  # (B, K, latent_size) - will be ignored
f = torch.randn(2, 16, 16, 512)  # (B, H, W, C)
z_out, f_out = wrapper(z, f)
print(f"   Latent in: {z.shape} -> Latent out: {z_out.shape} (unchanged)")
print(f"   Spatial in: {f.shape} -> Spatial out: {f_out.shape}")
assert z.shape == z_out.shape and f.shape == f_out.shape, "Shape mismatch!"
assert torch.equal(z, z_out), "Latent should pass through unchanged!"
print("   ✓ Passed")

# Test 3: No Latent version
print("\n3. Testing SelfAttentionNoLatent (no latent components):")
no_latent = SelfAttentionNoLatent(channels=512, num_heads=8)
f = torch.randn(2, 16, 16, 512)  # (B, H, W, C)
f_out = no_latent(f)
print(f"   Input: {f.shape} -> Output: {f_out.shape}")
assert f.shape == f_out.shape, "Shape mismatch!"
print("   ✓ Passed")

print("\n✅ All tests passed!")
