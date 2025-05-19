import torch
import torch.nn as nn
import math
from typing import Tuple

class GaussianMixture(nn.Module):
    def __init__(self, n_components=3, n_features=8):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        
        # Initialize parameters
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)
        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.logvars = nn.Parameter(torch.zeros(n_components, n_features))
        
    def forward(self, x):
        """Compute log probabilities for each component"""
        # x shape: [batch_size, n_features]
        batch_size = x.shape[0]
        
        # Reshape inputs for broadcasting
        x = x.unsqueeze(1)  # [batch_size, 1, n_features]
        means = self.means.unsqueeze(0)  # [1, n_components, n_features]
        logvars = self.logvars.unsqueeze(0)  # [1, n_components, n_features]
        
        # Compute log probabilities
        diff = x - means  # [batch_size, n_components, n_features]
        log_vars = logvars  # [1, n_components, n_features]
        
        # Compute log probability per dimension
        log_prob = -0.5 * (
            log_vars + 
            (diff ** 2 / torch.exp(log_vars)) + 
            math.log(2 * math.pi)
        )
        
        # Sum over features
        log_prob = log_prob.sum(-1)  # [batch_size, n_components]
        
        # Add log mixture weights
        log_prob += torch.log_softmax(self.weights, dim=0)
        
        return log_prob

class GMMAnomalyDetector:
    def __init__(self, n_components=5, n_features=8, device='cuda'):
        self.gmm = GaussianMixture(n_components, n_features).to(device)
        self.is_fitted = False
        self.device = device
        self.n_features = n_features
        
    def fit(self, features: torch.Tensor, n_epochs=100, lr=1e-3):
        """Train the GMM on the given features"""
        optimizer = torch.optim.Adam(self.gmm.parameters(), lr=lr)
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            log_probs = self.gmm(features)
            loss = -torch.logsumexp(log_probs, dim=1).mean()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gmm.parameters(), max_norm=1.0)
            optimizer.step()
            
        self.is_fitted = True
    
    def score_samples(self, features: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores for input features"""
        if not self.is_fitted:
            raise RuntimeError("GMM not fitted yet")
        log_probs = self.gmm(features)
        return -torch.logsumexp(log_probs, dim=1)

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
