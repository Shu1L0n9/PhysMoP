from os.path import join
from easydict import EasyDict as edict

hist_length = 25
pred_length = 25
total_length = hist_length + pred_length

dim = 63
motion_mlp = edict()
motion_mlp.hidden_dim = dim
motion_mlp.seq_len = hist_length
motion_mlp.with_normalization = True
motion_mlp.spatial_fc_only = False
motion_mlp.norm_axis = 'spatial'

motion_mlp.num_layers = 48

DATASET_FOLDERS = {
                    'H36M':  './dataset/data_processed/h36m_train_%d.pkl' % total_length,
                    'AMASS':  './dataset/data_processed/amass_train_%d.pkl' % total_length,
                    'PW3D':  './dataset/data_processed/pw3d_test_%d.pkl' % total_length,
                                  }

DATASET_VALID_PATH = {
                    # 'AMASS':  './dataset/data_processed/amass_train_%d_sub.pkl' % total_length,
                  }

DATASET_FOLDERS_TEST = {
                    'H36M': './dataset/data_processed/h36m_test_%d.pkl' % total_length,
                    'PW3D':  './dataset/data_processed/pw3d_test_%d.pkl' % total_length,
                    'AMASS':  './dataset/data_processed/amass_test_%d.pkl' % total_length,
                }

partition = [1.]

SMPL_MODEL_PATH = './dataset/smpl_official/processed_basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
SMPLH_N_PATH = './dataset/smpl_official/neutral/model.npz'
SMPLH_M_PATH = './dataset/smpl_official/male/model.npz'
SMPLH_F_PATH = './dataset/smpl_official/female/model.npz'

test_mode = 'AMASS' # 'H36M' 

# FNO Operator Configuration
# Model type: 'rollout' (original) or 'fno_operator' (new FNO-based)
model_kind = 'rollout'  # Default to original rollout mode

# FNO-specific parameters (used when model_kind == 'fno_operator')
fno_config = edict()
fno_config.d_model = 128           # Hidden dimension for FNO layers
fno_config.num_layers = 4          # Number of FNO spectral convolution layers
fno_config.modes = 16              # Number of Fourier modes to keep
fno_config.time_embed = 'fourier'  # Time positional encoding type
fno_config.time_embed_dim = 4      # Dimension for time embedding
fno_config.use_real_fft = True     # Use real FFT for efficiency
fno_config.normalize = True        # Normalize inputs/outputs
fno_config.padding_mode = 'reflect' # Padding for FFT
fno_config.dealias = True          # Apply dealiasing

# Physics regularizer parameters (for FNO mode)
physics_config = edict()
physics_config.enabled = True      # Enable physics consistency regularizer
physics_config.use_cholesky = True # Use Cholesky decomposition for M
physics_config.epsilon = 1e-6      # Numerical stability constant
physics_config.approx_derivatives = False  # Use stop-grad for derivatives

# Loss weights (for FNO mode)
loss_weights = edict()
loss_weights.lambda_data = 1.0     # Weight for data loss
loss_weights.lambda_physics = 0.1  # Weight for physics residual loss
loss_weights.lambda_smooth = 0.01  # Weight for smoothness loss
loss_weights.physics_warmup_epochs = 2  # Epochs for physics loss warmup

# Query sampling strategy
query_sampling = 'uniform'  # 'uniform' or 'random'


