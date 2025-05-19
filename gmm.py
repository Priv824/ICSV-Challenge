import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from typing import Tuple

class GMMAnomalyDetector:
    def __init__(self, n_components=3):
        self.gmm = GaussianMixture(n_components=n_components)
        self.is_fitted = False
        
    def fit(self, features: np.ndarray):
        self.gmm.fit(features)
        self.is_fitted = True
        
    def score_samples(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("GMM not fitted yet")
        return -self.gmm.score_samples(features)  # Higher score = more anomalous

def save_gmm(gmm: GMMAnomalyDetector, feature_means: np.ndarray, path: str):
    torch.save({
        'gmm': gmm,
        'feature_means': feature_means
    }, path)

def load_gmm(path: str) -> Tuple[GMMAnomalyDetector, np.ndarray]:
    data = torch.load(path)
    return data['gmm'], data['feature_means']