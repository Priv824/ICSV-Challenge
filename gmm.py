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
        self.cov_factor = nn.Parameter(torch.randn(n_components, n_features, n_features))
        
        # Diagonal covariance for numerical stability
        self.cov_diag = nn.Parameter(torch.ones(n_components, n_features))
        
    def forward(self, x):
        """Compute log probabilities for each component"""
        batch_size = x.shape[0]
        x = x.unsqueeze(1)  # [batch, 1, features]
        
        # Compute covariance matrices
        cov = torch.matmul(self.cov_factor, self.cov_factor.transpose(1, 2))
        cov = cov + torch.diag_embed(self.cov_diag)
        
        # Compute Mahalanobis distance
        diff = x - self.means.unsqueeze(0)  # [batch, components, features]
        cov_inv = torch.inverse(cov)  # [components, features, features]
        
        exponent = -0.5 * torch.einsum('bci,cij,bcj->bc', diff, cov_inv, diff)
        
        # Compute log probabilities
        log_det = torch.logdet(cov)  # [components]
        log_2pi = math.log(2 * math.pi)
        log_probs = exponent - 0.5 * (self.n_features * log_2pi + log_det.unsqueeze(0))
        
        # Weighted log probabilities
        weighted_log_probs = log_probs + torch.log(F.softmax(self.weights, dim=0)).unsqueeze(0)
        
        return weighted_log_probs

class GMMAnomalyDetector:
    def __init__(self, n_components=3, n_features=4, device='cuda'):
        self.gmm = GaussianMixture(n_components, n_features).to(device)
        self.is_fitted = False
        self.device = device
        self.n_features = n_features
        
    def fit(self, features: torch.Tensor, n_epochs=100, lr=1e-3):
        """Train GMM using expectation-maximization"""
        optimizer = torch.optim.Adam(self.gmm.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            # E-step: compute responsibilities
            log_probs = self.gmm(features)
            log_responsibilities = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
            responsibilities = torch.exp(log_responsibilities)
            
            # M-step: update parameters
            optimizer.zero_grad()
            
            # Negative log likelihood loss
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
