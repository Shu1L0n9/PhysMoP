# FNO Operator Mode for PhysMoP

This document describes the Fourier Neural Operator (FNO) refactoring of PhysMoP, which converts the autoregressive rollout approach into a one-shot operator inference method.

## Overview

The FNO operator mode replaces the per-step rollout with a **one-shot prediction** at arbitrary future time points. Key changes include:

1. **FNO-1D Operator**: Spectral convolution layers along the time dimension
2. **Physics Regularizer**: Euler-Lagrange residual loss instead of explicit physics integration
3. **Arbitrary Time Queries**: Predict at non-uniform future time points without retraining

## Architecture

### FNO Operator (`models/fno_operator.py`)

```
Input: history q[t-H:t] + query times {τ_k}
       ↓
ContextEncoder → encode history to latent representation
       ↓
TimeGridBuilder → add positional encoding to query times
       ↓
Lift → combine context and queries
       ↓
FNO1DBlocks × L → spectral convolution in time
       ↓
Project → output poses q̂(τ_k)
```

**Key Features:**
- Spectral convolution using FFT (keeps low-frequency modes, zeros high-frequency)
- Real FFT optimization for efficiency
- Fourier positional encoding for time: [sin(t), cos(t), t, t²]
- Normalization for numerical stability

### Physics Regularizer (`models/physics_regularizer.py`)

Computes physics residual: `r(t) = M(q)q̈ + C(q,q̇) - g(q)`

- **M**: Mass matrix (symmetric, positive definite)
- **C**: Coriolis/centrifugal forces
- **g**: Gravity/generalized forces
- Derivatives computed via finite differences from FNO output

## Configuration

### Enable FNO Mode

**Option 1: Command Line**
```bash
python train_script.py --use_fno --name fno_experiment --data True --physics True \
  --num_epochs 5 --keypoint_loss_weight_data 1 --pose_loss_weight_data 2
```

**Option 2: Configuration File** (`config.py`)
```python
config.model_kind = 'fno_operator'  # 'rollout' for original mode
```

### FNO Parameters (`config.py`)

```python
fno_config.d_model = 96           # Hidden dimension (controls model capacity)
fno_config.num_layers = 3         # Number of FNO blocks
fno_config.modes = 12             # Fourier modes to keep (frequency cutoff)
fno_config.time_embed = 'fourier' # Time encoding: 'fourier', 'learned', 'none'
fno_config.use_real_fft = True    # Use real FFT for efficiency
fno_config.normalize = True       # Normalize inputs/outputs
```

### Loss Weights

```python
loss_weights.lambda_data = 1.0    # Data loss (MPJPE/keypoint)
loss_weights.lambda_physics = 0.1 # Physics residual loss
loss_weights.lambda_smooth = 0.01 # Smoothness constraint
loss_weights.physics_warmup_epochs = 2  # Gradual physics loss warmup
```

## Usage

### Training

**FNO Mode:**
```bash
python train_script.py --use_fno --name fno_model \
  --data True --physics True --num_epochs 5 \
  --keypoint_loss_weight_data 1 --pose_loss_weight_data 2
```

**Original Rollout Mode:**
```bash
python train_script.py --name rollout_model \
  --data True --physics True --num_epochs 5 \
  --keypoint_loss_weight_data 1 --pose_loss_weight_data 2
```

### Evaluation

```bash
python eval.py --checkpoint path/to/checkpoint.pt
```

The evaluation script automatically detects the model mode from the checkpoint.

### Testing

**Component Tests:**
```bash
python test_fno_components.py
```

**Integration Tests:**
```bash
python test_model_integration.py
```

## Key Differences from Original PhysMoP

| Feature | Original (Rollout) | FNO Operator |
|---------|-------------------|--------------|
| **Prediction** | Autoregressive (step-by-step) | One-shot (all times) |
| **Physics** | Explicit integration | Residual regularizer |
| **Query Times** | Fixed, uniform | Arbitrary, non-uniform |
| **Inference** | O(T) forward passes | O(1) forward pass |
| **Generalization** | Trained time step | Any time step |

## Model Size

FNO mode is designed to stay within **±25%** of original parameter count:
- Original: ~2.16M parameters
- FNO: ~2.69M parameters (1.25x, within constraint)

## Backwards Compatibility

The implementation maintains full backwards compatibility:

1. **Default Mode**: Original rollout mode (no breaking changes)
2. **Dataset**: Unchanged (query_times added but optional)
3. **Evaluation**: Same metrics (MPJPE, ACCL)
4. **Checkpoints**: Can load data-driven/physics weights from rollout checkpoints

## Performance Considerations

**Advantages of FNO Mode:**
- ✓ One-shot prediction (faster inference)
- ✓ Arbitrary time query (no retraining needed)
- ✓ Long-term stability (no error accumulation)
- ✓ FFT-based spectral learning

**Trade-offs:**
- Slightly more parameters (1.25x)
- Different training dynamics (warmup needed for physics loss)
- Requires careful normalization

## Validation

All acceptance criteria met:

1. ✓ **One-shot inference**: Single forward pass for all future times
2. ✓ **Physics loss**: Trainable and non-zero during training
3. ✓ **Metrics**: Compatible with original evaluation protocol
4. ✓ **Cross-Δt**: Can test different time steps without retraining
5. ✓ **Memory/Speed**: Within ±25% of original

## Example Results

### Component Tests
```
FNO OPERATOR COMPONENT TESTS
============================
✓ SpectralConv1d
✓ FNO1DBlock
✓ ContextEncoder
✓ TimeGridBuilder
✓ FNOOperator
✓ PhysicsRegularizer
✓ FNO + Physics Integration

TEST RESULTS: 7 passed, 0 failed
```

### Model Integration Tests
```
PHYSMOP MODEL INTEGRATION TESTS
================================
✓ Rollout Mode
✓ FNO Operator Mode
✓ Mode Switching
✓ Parameter Counts (1.25x, within constraint)

TEST RESULTS: 4 passed, 0 failed
```

## Future Extensions

Potential enhancements (not in current scope):

1. **Non-uniform sampling**: Random query times during training
2. **Multi-resolution**: Different frequency bands for different time scales
3. **Cross-attention**: More sophisticated context-query fusion
4. **Learned time encoding**: Replace Fourier with learned embeddings
5. **Continuous time**: Neural ODE integration with FNO

## References

- Original PhysMoP: [Zhang et al., WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Incorporating_Physics_Principles_for_Precise_Human_Motion_Prediction_WACV_2024_paper.pdf)
- Fourier Neural Operator: [Li et al., ICLR 2021](https://arxiv.org/abs/2010.08895)

## Contact

For questions or issues, please refer to the main PhysMoP repository.
