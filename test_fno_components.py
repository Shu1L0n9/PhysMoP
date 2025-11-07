"""
Simple test script for FNO operator components
Tests basic functionality without requiring full dataset
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from models.fno_operator import FNOOperator, SpectralConv1d, FNO1DBlock, ContextEncoder, TimeGridBuilder
from models.physics_regularizer import PhysicsRegularizer
import config

def test_spectral_conv1d():
    """Test SpectralConv1d layer"""
    print("\n=== Testing SpectralConv1d ===")
    
    batch_size = 4
    channels = 32
    time_steps = 25
    modes = 8
    
    layer = SpectralConv1d(channels, channels, modes, use_real_fft=True)
    x = torch.randn(batch_size, channels, time_steps)
    
    out = layer(x)
    
    assert out.shape == (batch_size, channels, time_steps), f"Expected shape {(batch_size, channels, time_steps)}, got {out.shape}"
    print(f"✓ SpectralConv1d output shape: {out.shape}")
    print(f"✓ Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    
    return True

def test_fno1d_block():
    """Test FNO1DBlock"""
    print("\n=== Testing FNO1DBlock ===")
    
    batch_size = 4
    channels = 64
    time_steps = 25
    modes = 12
    
    block = FNO1DBlock(channels, modes, use_real_fft=True)
    x = torch.randn(batch_size, channels, time_steps)
    
    out = block(x)
    
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    print(f"✓ FNO1DBlock output shape: {out.shape}")
    print(f"✓ Has residual connection: {torch.allclose(out, x, atol=1.0) == False}")
    
    return True

def test_context_encoder():
    """Test ContextEncoder"""
    print("\n=== Testing ContextEncoder ===")
    
    batch_size = 4
    hist_length = 25
    input_dim = 63
    hidden_dim = 128
    
    encoder = ContextEncoder(input_dim, hidden_dim, hist_length, normalize=True)
    hist = torch.randn(batch_size, hist_length, input_dim)
    
    out = encoder(hist)
    
    assert out.shape == (batch_size, hist_length, hidden_dim), f"Expected shape {(batch_size, hist_length, hidden_dim)}, got {out.shape}"
    print(f"✓ ContextEncoder output shape: {out.shape}")
    print(f"✓ Input stats - mean: {encoder.input_mean[:5]}")
    print(f"✓ Input stats - std: {encoder.input_std[:5]}")
    
    return True

def test_time_grid_builder():
    """Test TimeGridBuilder"""
    print("\n=== Testing TimeGridBuilder ===")
    
    batch_size = 4
    K = 25
    
    builder = TimeGridBuilder(embed_type='fourier', embed_dim=4)
    query_times = torch.arange(25, 50, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    
    out = builder(query_times)
    
    assert out.shape == (batch_size, K, 4), f"Expected shape {(batch_size, K, 4)}, got {out.shape}"
    print(f"✓ TimeGridBuilder output shape: {out.shape}")
    print(f"✓ Time encoding sample: {out[0, 0, :]}")
    
    return True

def test_fno_operator():
    """Test full FNO operator"""
    print("\n=== Testing FNOOperator ===")
    
    batch_size = 4
    hist_length = 25
    query_length = 25
    input_dim = 63
    hidden_dim = 128
    num_layers = 2
    modes = 12
    
    operator = FNOOperator(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        modes=modes,
        hist_length=hist_length,
        time_embed='fourier',
        time_embed_dim=4,
        use_real_fft=True,
        normalize=True
    )
    
    hist = torch.randn(batch_size, hist_length, input_dim)
    query_times = torch.arange(hist_length, hist_length + query_length, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    
    out = operator(hist, query_times)
    
    assert out.shape == (batch_size, query_length, input_dim), f"Expected shape {(batch_size, query_length, input_dim)}, got {out.shape}"
    print(f"✓ FNOOperator output shape: {out.shape}")
    print(f"✓ Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    
    # Test gradient flow
    loss = out.mean()
    loss.backward()
    has_grad = any(p.grad is not None for p in operator.parameters())
    print(f"✓ Gradients computed: {has_grad}")
    
    return True

def test_physics_regularizer():
    """Test PhysicsRegularizer"""
    print("\n=== Testing PhysicsRegularizer ===")
    
    batch_size = 4
    K = 25
    dim = 63
    feature_dim = dim * 4
    
    regularizer = PhysicsRegularizer(
        dim=dim,
        use_cholesky=True,
        epsilon=1e-6,
        approx_derivatives=False
    )
    
    q = torch.randn(batch_size, K, dim)
    query_times = torch.arange(25, 50, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    motion_feats = torch.randn(batch_size, feature_dim)
    
    loss, residuals = regularizer(q, query_times, motion_feats, dt=1/25)
    
    assert isinstance(loss.item(), float), f"Expected scalar loss, got {type(loss)}"
    assert residuals.shape == (batch_size, K, dim), f"Expected residuals shape {(batch_size, K, dim)}, got {residuals.shape}"
    print(f"✓ Physics loss: {loss.item():.6f}")
    print(f"✓ Residuals shape: {residuals.shape}")
    print(f"✓ Residuals range: [{residuals.min().item():.3f}, {residuals.max().item():.3f}]")
    
    # Test gradient flow
    loss.backward()
    has_grad = any(p.grad is not None for p in regularizer.parameters())
    print(f"✓ Gradients computed: {has_grad}")
    
    return True

def test_integration():
    """Test FNO + Physics together"""
    print("\n=== Testing FNO + Physics Integration ===")
    
    batch_size = 2
    hist_length = 25
    query_length = 25
    input_dim = 63
    
    # Create operator and regularizer
    operator = FNOOperator(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        modes=8,
        hist_length=hist_length
    )
    
    regularizer = PhysicsRegularizer(dim=input_dim)
    
    # Forward pass
    hist = torch.randn(batch_size, hist_length, input_dim)
    query_times = torch.arange(hist_length, hist_length + query_length, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    
    q_pred = operator(hist, query_times)
    motion_feats = hist.mean(dim=1).repeat(1, 4)  # Simple context
    
    loss_physics, residuals = regularizer(q_pred, query_times, motion_feats)
    loss_data = (q_pred ** 2).mean()
    
    total_loss = loss_data + 0.1 * loss_physics
    
    print(f"✓ Data loss: {loss_data.item():.6f}")
    print(f"✓ Physics loss: {loss_physics.item():.6f}")
    print(f"✓ Total loss: {total_loss.item():.6f}")
    
    # Backward pass
    total_loss.backward()
    has_grad = any(p.grad is not None for p in operator.parameters())
    print(f"✓ Gradients computed: {has_grad}")
    
    return True

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("FNO OPERATOR COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        test_spectral_conv1d,
        test_fno1d_block,
        test_context_encoder,
        test_time_grid_builder,
        test_fno_operator,
        test_physics_regularizer,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
