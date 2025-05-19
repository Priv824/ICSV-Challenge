import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

class GaussianMixture(nn.Module):
    def __init__(self, n_components=3, n_features=4):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        
        # Initialize parameters
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)
        self.means = nn.Parameter(torch.randn(n_components, n_features))
        
        # Use diagonal covariance for stability
        self.logvars = nn.Parameter(torch.zeros(n_components, n_features))
        
    def forward(self, x):
        """Compute log probabilities for each component"""
        # x shape: [batch_size, n_features]
        batch_size = x.shape[0]
        
        # Expand dimensions for broadcasting
        x = x.unsqueeze(1)  # [batch_size, 1, n_features]
        means = self.means.unsqueeze(0)  # [1, n_components, n_features]
        
        # Compute squared Mahalanobis distance with diagonal covariance
        diff = x - means  # [batch_size, n_components, n_features]
        variances = torch.exp(self.logvars)
        inv_variances = 1.0 / (variances + 1e-6)  # Add small epsilon for stability
        
        # Compute exponent term: -0.5 * Σ((x-μ)²/σ²)
        exponent = -0.5 * torch.sum(diff.pow(2) * inv_variances.unsqueeze(0), dim=-1)
        
        # Compute log normalization term: -0.5 * (n_features*log(2π) + Σlog(σ²))
        log_normalization = -0.5 * (self.n_features * math.log(2 * math.pi) + 
                            torch.sum(self.logvars, dim=-1))
        log_normalization = log_normalization.unsqueeze(0)  # [1, n_components]
        
        # Compute log probabilities
        log_probs = exponent + log_normalization
        
        # Add log weights
        weighted_log_probs = log_probs + torch.log_softmax(self.weights, dim=0).unsqueeze(0)
        
        return weighted_log_probs
    

class GMMAnomalyDetector:
    def __init__(self, n_components=3, n_features=4, device='cuda'):
        self.gmm = GaussianMixture(n_components, n_features).to(device)
        self.is_fitted = False
        self.device = device
        self.n_features = n_features
        
    def fit(self, features: torch.Tensor, n_epochs=100, lr=1e-3):
        optimizer = torch.optim.Adam(self.gmm.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            log_probs = self.gmm(features)
            loss = -torch.logsumexp(log_probs, dim=1).mean()
            loss.backward()
            optimizer.step()
            
        self.is_fitted = True
        
    def score_samples(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("GMM not fitted yet")
        log_probs = self.gmm(features)
        return -torch.logsumexp(log_probs, dim=1)  # Higher score = more anomalous
    
def save_gmm(gmm: GMMAnomalyDetector, feature_means: torch.Tensor, path: str):
    torch.save({
        'gmm_state_dict': gmm.gmm.state_dict(),
        'feature_means': feature_means.cpu(),
        'n_components': gmm.gmm.n_components,
        'n_features': gmm.n_features,
        'device': gmm.device
    }, path)

def load_gmm(path: str, device='cuda') -> Tuple[GMMAnomalyDetector, torch.Tensor]:
    data = torch.load(path, map_location=device)
    detector = GMMAnomalyDetector(
        n_components=data['n_components'],
        n_features=data['n_features'],
        device=data['device']
    )
    detector.gmm.load_state_dict(data['gmm_state_dict'])
    detector.is_fitted = True
    feature_means = data['feature_means'].to(device)
    return detector, feature_means
