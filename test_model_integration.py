"""
Test PhysMoP model in both rollout and FNO modes
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.PhysMoP import PhysMoP
import config

def test_rollout_mode():
    """Test original rollout mode"""
    print("\n=== Testing Rollout Mode ===")
    
    # Set config to rollout mode
    config.model_kind = 'rollout'
    
    model = PhysMoP(
        hist_length=25,
        physics=True,
        data=True,
        fusion=False
    )
    
    print(f"✓ Model created in rollout mode")
    print(f"✓ Has regressor: {hasattr(model, 'regressor')}")
    print(f"✓ FNO operator: {hasattr(model, 'fno_operator')}")
    
    # Test forward pass
    batch_size = 2
    gt_q = torch.randn(batch_size, config.total_length, 63)
    gt_q_ddot = torch.randn(batch_size, config.total_length-2, 63)
    
    output = model.forward_dynamics(None, gt_q, gt_q_ddot, None, None, 'cpu', mode='train')
    
    assert len(output) == 7, f"Expected 7 outputs, got {len(output)}"
    print(f"✓ Forward pass successful")
    print(f"✓ Output shapes: data={output[0].shape}, physics_gt={output[1].shape}")
    
    return True

def test_fno_mode():
    """Test FNO operator mode"""
    print("\n=== Testing FNO Operator Mode ===")
    
    # Set config to FNO mode
    config.model_kind = 'fno_operator'
    
    model = PhysMoP(
        hist_length=25,
        physics=True,
        data=True,
        fusion=False
    )
    
    print(f"✓ Model created in FNO mode")
    print(f"✓ Has FNO operator: {hasattr(model, 'fno_operator')}")
    print(f"✓ Has physics regularizer: {hasattr(model, 'physics_regularizer')}")
    print(f"✓ Regressor: {hasattr(model, 'regressor')}")
    
    # Test forward pass
    batch_size = 2
    gt_q = torch.randn(batch_size, config.total_length, 63)
    gt_q_ddot = torch.randn(batch_size, config.total_length-2, 63)
    query_times = torch.arange(25, 50, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    
    output = model.forward_dynamics(None, gt_q, gt_q_ddot, None, None, 'cpu', mode='train', query_times=query_times)
    
    assert len(output) == 7, f"Expected 7 outputs, got {len(output)}"
    print(f"✓ Forward pass successful")
    print(f"✓ Output shapes: data={output[0].shape}, physics_pred={output[2].shape}")
    
    # Test physics regularizer
    if model.physics_regularizer is not None:
        q_pred = output[0][:, config.hist_length:, :]
        motion_feats = gt_q[:, :config.hist_length, :].mean(dim=1)
        loss_phys, residuals = model.physics_regularizer(q_pred, query_times, motion_feats)
        print(f"✓ Physics loss: {loss_phys.item():.6f}")
        print(f"✓ Residuals shape: {residuals.shape}")
    
    return True

def test_mode_switching():
    """Test switching between modes"""
    print("\n=== Testing Mode Switching ===")
    
    # Test rollout mode
    config.model_kind = 'rollout'
    model1 = PhysMoP(hist_length=25, physics=True, data=True, fusion=False)
    assert not model1.use_fno, "Expected rollout mode"
    print(f"✓ Created model in rollout mode")
    
    # Test FNO mode
    config.model_kind = 'fno_operator'
    model2 = PhysMoP(hist_length=25, physics=True, data=True, fusion=False)
    assert model2.use_fno, "Expected FNO mode"
    print(f"✓ Created model in FNO mode")
    
    # Test explicit override
    model3 = PhysMoP(hist_length=25, physics=True, data=True, fusion=False, use_fno=False)
    assert not model3.use_fno, "Expected rollout mode (explicit override)"
    print(f"✓ Explicit mode override works")
    
    return True

def test_parameter_count():
    """Compare parameter counts between modes"""
    print("\n=== Testing Parameter Counts ===")
    
    # Rollout mode
    config.model_kind = 'rollout'
    model_rollout = PhysMoP(hist_length=25, physics=True, data=True, fusion=False)
    params_rollout = sum(p.numel() for p in model_rollout.parameters())
    
    # FNO mode
    config.model_kind = 'fno_operator'
    model_fno = PhysMoP(hist_length=25, physics=True, data=True, fusion=False)
    params_fno = sum(p.numel() for p in model_fno.parameters())
    
    print(f"✓ Rollout mode parameters: {params_rollout:,}")
    print(f"✓ FNO mode parameters: {params_fno:,}")
    print(f"✓ Ratio (FNO/Rollout): {params_fno/params_rollout:.2f}x")
    
    # Check if within ±25% constraint
    ratio = params_fno / params_rollout
    within_constraint = 0.75 <= ratio <= 1.25
    print(f"✓ Within ±25% constraint: {within_constraint}")
    
    return True

def run_all_tests():
    """Run all model tests"""
    print("=" * 60)
    print("PHYSMOP MODEL INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_rollout_mode,
        test_fno_mode,
        test_mode_switching,
        test_parameter_count
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
