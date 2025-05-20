import torch
import numpy as np
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans

class GaussianMixture(torch.nn.Module):
    def __init__(self, n_components=5, n_features=4, reg_covar=1e-6):
        super().__init__()
        self.n_components = n_components  # Increased from original 3
        self.n_features = n_features
        self.reg_covar = reg_covar

        # Parameters for EM
        self.weights = torch.nn.Parameter(torch.ones(n_components) / n_components)
        self.means = torch.nn.Parameter(torch.randn(n_components, n_features))
        self.covariances = torch.nn.Parameter(torch.stack([torch.eye(n_features) 
                                                          for _ in range(n_components)]))
    
    def _initialize_parameters(self, X):
        """Improved initialization using k-means"""
        kmeans = KMeans(n_clusters=self.n_components).fit(X.cpu().numpy())
        self.means.data = torch.tensor(kmeans.cluster_centers_, device=X.device)
        for k in range(self.n_components):
            cluster_points = X[kmeans.labels_ == k].to(X.device)  # Move to the same device
            cov = torch.cov(cluster_points.T) + self.reg_covar * torch.eye(self.n_features, device=X.device)  # Ensure identity matrix is on the same device
            self.covariances.data[k] = cov

    def _e_step(self, X):
        """Expectation step with full covariance support"""
        log_prob = torch.zeros(X.size(0), self.n_components, device=X.device)
        
        for k in range(self.n_components):
            diff = X - self.means[k]
            cov = self.covariances[k] + self.reg_covar * torch.eye(self.n_features, device=X.device)
            
            # Calculate multivariate normal log probability
            L = torch.linalg.cholesky(cov)
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))
            mahalanobis = torch.linalg.solve_triangular(L, diff.T, upper=False).pow(2).sum(dim=0)
            
            log_prob[:, k] = -0.5 * (self.n_features * np.log(2 * np.pi) + 
                                    log_det + mahalanobis)
            
        log_weights = torch.log(self.weights + 1e-10)
        log_resp = log_prob + log_weights
        log_resp -= torch.logsumexp(log_resp, dim=1, keepdim=True)
        return torch.exp(log_resp)

    def _m_step(self, X, resp):
        """Maximization step"""
        Nk = resp.sum(dim=0)
        
        # Update weights
        self.weights.data = Nk / X.size(0)
        
        # Update means
        self.means.data = (resp.T @ X) / Nk[:, None]
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            cov = (resp[:, k] * diff.T) @ diff / Nk[k]
            self.covariances.data[k] = cov + self.reg_covar * torch.eye(self.n_features, device=X.device)

    def fit(self, X, max_iter=100, tol=1e-4):
        """Full EM algorithm implementation"""
        self._initialize_parameters(X)
        prev_lower_bound = -np.inf
        
        for iteration in range(max_iter):
            # E-step
            resp = self._e_step(X)
            
            # M-step
            self._m_step(X, resp)
            
            # Check convergence
            current_log_prob = self.score_samples(X).mean()
            if abs(current_log_prob - prev_lower_bound) < tol:
                break
            prev_lower_bound = current_log_prob

    def score_samples(self, X):
        """Calculate log probabilities"""
        log_prob = torch.zeros(X.size(0), self.n_components, device=X.device)
        for k in range(self.n_components):
            diff = X - self.means[k]
            cov = self.covariances[k] + self.reg_covar * torch.eye(self.n_features, device=X.device)
            try:
                L = torch.linalg.cholesky(cov)
            except:
                L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(self.n_features, device=X.device))
                
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))
            mahalanobis = torch.linalg.solve_triangular(L, diff.T, upper=False).pow(2).sum(dim=0)
            
            log_prob[:, k] = -0.5 * (self.n_features * np.log(2 * np.pi) + 
                                    log_det + mahalanobis) + torch.log(self.weights[k])
            
        return torch.logsumexp(log_prob, dim=1)

class GMMAnomalyDetector:
    def __init__(self, n_components=5, n_features=4, device='cuda'):
        self.gmm = GaussianMixture(n_components, n_features).to(device)
        self.device = device
        
    def fit(self, frames: torch.Tensor, max_iter=100, tol=1e-4):
        """Train using EM algorithm"""
        self.gmm.fit(frames.to(self.device), max_iter=max_iter, tol=tol)
    
    def score_file(self, frames: torch.Tensor) -> float:
        """Score an audio file using negative log-likelihood"""
        with torch.no_grad():
            log_probs = self.gmm.score_samples(frames.to(self.device))
            return -log_probs.mean().item()