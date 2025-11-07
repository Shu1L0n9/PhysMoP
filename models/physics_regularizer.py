"""
Physics Regularizer for FNO-based PhysMoP
Converts the physics forward step into a physics-consistency regularizer
by computing Euler-Lagrange residuals on FNO operator outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsRegularizer(nn.Module):
    """
    Physics-based regularization using Euler-Lagrange equations.
    Computes residual: r(t) = M(q)q̈ + C(q,q̇) - g(q)
    """
    
    def __init__(
        self,
        dim=63,
        hidden_dim=256,
        use_cholesky=True,
        epsilon=1e-6,
        approx_derivatives=False
    ):
        super(PhysicsRegularizer, self).__init__()
        
        self.dim = dim
        self.use_cholesky = use_cholesky
        self.epsilon = epsilon
        self.approx_derivatives = approx_derivatives
        
        # Reuse MLP architectures from original physics branch
        # These networks estimate M, C, g from state
        
        # Input: concatenated features (from history encoding + current state)
        feature_dim = dim * 4  # Adjust based on input features
        
        # Mass matrix M network (outputs Cholesky factor L where M = LL^T + εI)
        M_layer_size = [feature_dim, 512, 512, dim * 32]
        self.M_net_FC1 = nn.Linear(M_layer_size[0], M_layer_size[1])
        self.M_net_relu1 = nn.ReLU()
        self.M_net_FC2 = nn.Linear(M_layer_size[1], M_layer_size[2])
        self.M_net_relu2 = nn.ReLU()
        self.M_net_FC3 = nn.Linear(M_layer_size[2], M_layer_size[3])
        
        # Coriolis/centrifugal term C network
        C_layer_size = [feature_dim, 512, 256, dim]
        self.C_net_FC1 = nn.Linear(C_layer_size[0], C_layer_size[1])
        self.C_net_relu1 = nn.ReLU()
        self.C_net_FC2 = nn.Linear(C_layer_size[1], C_layer_size[2])
        self.C_net_relu2 = nn.ReLU()
        self.C_net_FC3 = nn.Linear(C_layer_size[2], C_layer_size[3])
        
        # Gravity/generalized forces g network (actuation)
        g_layer_size = [feature_dim, 512, 256, dim]
        self.g_net_FC1 = nn.Linear(g_layer_size[0], g_layer_size[1])
        self.g_net_relu1 = nn.ReLU()
        self.g_net_FC2 = nn.Linear(g_layer_size[1], g_layer_size[2])
        self.g_net_relu2 = nn.ReLU()
        self.g_net_FC3 = nn.Linear(g_layer_size[2], g_layer_size[3])
    
    def estimate_M_inv(self, features):
        """
        Estimate inverse mass matrix M^{-1} from features.
        Uses symmetric matrix construction to ensure physical validity.
        
        Args:
            features: (batch, feature_dim)
        Returns:
            M_inv: (batch, dim, dim) - inverse mass matrix
        """
        batch_size = features.shape[0]
        
        # Forward through M network
        x = self.M_net_FC1(features)
        x = self.M_net_relu1(x)
        x = self.M_net_FC2(x)
        x = self.M_net_relu2(x)
        M_vector = self.M_net_FC3(x)  # (batch, dim*32)
        
        # Construct symmetric matrix (lower triangular + transpose)
        M_inv = torch.zeros((batch_size, self.dim, self.dim), 
                           dtype=features.dtype, device=features.device)
        
        # Fill lower triangular part
        tril_indices = torch.tril_indices(row=self.dim, col=self.dim, offset=0)
        M_inv[:, tril_indices[0], tril_indices[1]] = M_vector[:, :len(tril_indices[0])]
        
        # Make symmetric
        M_inv[:, tril_indices[1], tril_indices[0]] = M_vector[:, :len(tril_indices[0])]
        
        # Add small epsilon for numerical stability
        M_inv = M_inv + self.epsilon * torch.eye(self.dim, device=features.device).unsqueeze(0)
        
        return M_inv
    
    def estimate_C(self, features):
        """
        Estimate Coriolis/centrifugal forces from features.
        
        Args:
            features: (batch, feature_dim)
        Returns:
            C: (batch, dim) - Coriolis forces
        """
        x = self.C_net_FC1(features)
        x = self.C_net_relu1(x)
        x = self.C_net_FC2(x)
        x = self.C_net_relu2(x)
        C = self.C_net_FC3(x)
        
        return C
    
    def estimate_g(self, features):
        """
        Estimate gravity/generalized forces from features.
        
        Args:
            features: (batch, feature_dim)
        Returns:
            g: (batch, dim) - generalized forces
        """
        x = self.g_net_FC1(features)
        x = self.g_net_relu1(x)
        x = self.g_net_FC2(x)
        x = self.g_net_relu2(x)
        g = self.g_net_FC3(x)
        
        return g
    
    def compute_derivatives(self, q, query_times, dt):
        """
        Compute time derivatives of q using finite differences or autodiff.
        
        Args:
            q: (batch, K, dim) - poses at query times
            query_times: (batch, K) - query time points
            dt: time step
        Returns:
            q_dot: (batch, K, dim) - velocities
            q_ddot: (batch, K, dim) - accelerations
        """
        batch_size, K, dim = q.shape
        
        if self.approx_derivatives:
            # Use finite differences (stop gradient for stability)
            with torch.no_grad():
                # Central differences for interior points
                q_dot = torch.zeros_like(q)
                q_ddot = torch.zeros_like(q)
                
                # Forward difference for first point
                q_dot[:, 0] = (q[:, 1] - q[:, 0]) / dt
                
                # Central difference for interior
                if K > 2:
                    q_dot[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2 * dt)
                
                # Backward difference for last point
                if K > 1:
                    q_dot[:, -1] = (q[:, -1] - q[:, -2]) / dt
                
                # Second derivative (acceleration)
                if K > 2:
                    q_ddot[:, 1:-1] = (q[:, 2:] - 2*q[:, 1:-1] + q[:, :-2]) / (dt**2)
        else:
            # Use autodiff (creates computational graph)
            # This allows gradient to flow through time derivatives
            q_dot = torch.zeros_like(q)
            q_ddot = torch.zeros_like(q)
            
            # Compute derivatives using finite differences
            # Forward difference for first point
            q_dot[:, 0] = (q[:, 1] - q[:, 0]) / dt
            
            # Central difference for interior
            if K > 2:
                q_dot[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2 * dt)
            
            # Backward difference for last point
            if K > 1:
                q_dot[:, -1] = (q[:, -1] - q[:, -2]) / dt
            
            # Second derivative
            if K > 2:
                q_ddot[:, 1:-1] = (q[:, 2:] - 2*q[:, 1:-1] + q[:, :-2]) / (dt**2)
        
        return q_dot, q_ddot
    
    def forward(self, q, query_times, motion_feats_all, dt=1/25):
        """
        Compute physics residual loss.
        
        Args:
            q: (batch, K, dim) - predicted poses at query times
            query_times: (batch, K) - query time points
            motion_feats_all: (batch, feature_dim) - encoded features from history
            dt: time step in seconds
        Returns:
            loss_physics: scalar - physics residual loss
            residuals: (batch, K, dim) - residuals at each time point
        """
        batch_size, K, dim = q.shape
        
        # Compute velocities and accelerations
        q_dot, q_ddot = self.compute_derivatives(q, query_times, dt)
        
        # Reshape for processing
        q_flat = q.reshape(-1, dim)  # (B*K, dim)
        q_dot_flat = q_dot.reshape(-1, dim)
        q_ddot_flat = q_ddot.reshape(-1, dim)
        
        # Prepare features by concatenating state information
        # Expand motion_feats_all to all time points
        motion_feats_expanded = motion_feats_all.unsqueeze(1).expand(-1, K, -1)
        motion_feats_flat = motion_feats_expanded.reshape(-1, motion_feats_all.shape[-1])
        
        # Concatenate current state to create full feature vector
        # Expected feature_dim = dim * 4 (from original implementation)
        # We have: motion_feats (varies), q, q_dot, q_ddot (each dim)
        # Pad motion_feats_flat or concatenate with state
        if motion_feats_flat.shape[-1] < self.dim * 4:
            # Concatenate with current state info to reach expected feature_dim
            features = torch.cat([motion_feats_flat, q_flat, q_dot_flat, q_ddot_flat], dim=-1)
            # If still not enough, pad or use first dim*4 dimensions
            if features.shape[-1] > self.dim * 4:
                features = features[:, :self.dim * 4]
            elif features.shape[-1] < self.dim * 4:
                # Pad with zeros
                padding = torch.zeros(features.shape[0], self.dim * 4 - features.shape[-1], 
                                     device=features.device, dtype=features.dtype)
                features = torch.cat([features, padding], dim=-1)
        else:
            features = motion_feats_flat[:, :self.dim * 4]
        
        # Estimate physics terms
        M_inv = self.estimate_M_inv(features).reshape(batch_size * K, dim, dim)
        C = self.estimate_C(features)  # (B*K, dim)
        g = self.estimate_g(features)  # (B*K, dim)
        
        # Compute physics residual: M^{-1}(g - C) - q̈
        # Or equivalently: M*q̈ + C - g (but we have M^{-1})
        # So: r = q̈ - M^{-1}(g - C)
        residual = q_ddot_flat - (M_inv @ (g - C).unsqueeze(-1)).squeeze(-1)
        
        # Reshape back
        residuals = residual.reshape(batch_size, K, dim)
        
        # Compute loss (mean squared residual)
        loss_physics = (residuals ** 2).mean()
        
        return loss_physics, residuals
