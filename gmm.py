import torch
import torch.nn as nn
import math

class GaussianMixture(nn.Module):
    def __init__(self, n_components=3, n_features=4):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)
        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.logvars = nn.Parameter(torch.zeros(n_components, n_features))
        
    def forward(self, x):
        # x shape: [n_frames, 4]
        x = x.unsqueeze(1)  # [n_frames, 1, 4]
        means = self.means.unsqueeze(0)  # [1, n_components, 4]
        logvars = self.logvars.unsqueeze(0)
        
        # Gaussian PDF
        diff = x - means
        prec = torch.exp(-logvars)
        log_prob = -0.5 * (logvars + (diff**2) * prec + math.log(2*math.pi))
        log_prob = log_prob.sum(-1)  # [n_frames, n_components]
        
        return log_prob + torch.log_softmax(self.weights, dim=0)

class GMMAnomalyDetector:
    def __init__(self, n_components=3, n_features=4, device='cuda'):
        self.gmm = GaussianMixture(n_components, n_features).to(device)
        self.device = device
        
    def fit(self, frames: torch.Tensor, n_epochs=100, lr=1e-3):
        """Train on frame-level features"""
        optimizer = torch.optim.Adam(self.gmm.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            log_probs = self.gmm(frames.to(self.device))
            loss = -torch.logsumexp(log_probs, dim=1).mean()
            loss.backward()
            optimizer.step()
    
    def score_file(self, frames: torch.Tensor) -> float:
        """Score an audio file by averaging frame scores"""
        with torch.no_grad():
            log_probs = self.gmm(frames.to(self.device))
            return -torch.logsumexp(log_probs, dim=1).mean().item()