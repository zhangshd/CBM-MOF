"""
Power Transformer Normalizer
Simple data transformer to convert long-tail distributions to near-normal distributions
Author: zhangshd
Date: October 10, 2025
"""

import torch
import numpy as np
from sklearn.preprocessing import PowerTransformer
from typing import Optional, Union, Literal
import warnings


class PowerTransformerNormalizer(object):
    """
    Simple normalizer using sklearn PowerTransformer for data preprocessing.
    
    Core functionality:
    1. Fit sklearn PowerTransformer to transform long-tail data to near-normal
    2. Store transformation parameters and use sklearn for all transformations
    3. Provide simple norm/denorm interface for training and inference
    
    This design prioritizes simplicity and reliability over GPU acceleration.
    """

    def __init__(
        self,
        method: Literal['box-cox', 'yeo-johnson'] = 'yeo-johnson',
        remove_value: Optional[float] = None,
        copy: bool = True
    ):
        """
        Initialize Power Transformer Normalizer.
        
        Args:
            method: Transformation method ('box-cox' or 'yeo-johnson')
            remove_value: Value to remove from data before fitting
            copy: Whether to copy data during transformation
        """
        super(PowerTransformerNormalizer, self).__init__()
        
        self.name = 'power_transformer'
        self.method = method
        self.remove_value = remove_value
        self.copy = copy
        self.device = torch.device('cpu')
        
        # sklearn transformer for fitting and transformation
        self._sklearn_transformer = PowerTransformer(
            method=method,
            standardize=True,  # Let sklearn handle standardization
            copy=copy
        )
        
        # Parameters for compatibility
        self.lambdas_ = None
        self.mean_ = None
        self.std_ = None
        self.scale_factor_ = 1.0  # Scaling factor for compatibility with existing models
        self._fitted = False
        
    def fit(self, tensor: torch.Tensor) -> 'PowerTransformerNormalizer':
        """
        Fit the power transformer to the data.
        
        Args:
            tensor: Input tensor to fit the transformer
            
        Returns:
            Self for method chaining
        """
        # Handle special case for classification (predefined values)
        if len(tensor) == 3 and torch.allclose(tensor, torch.tensor([-1., 0., 1.])):
            self.lambdas_ = torch.tensor([1.0])
            self.mean_ = 0.0
            self.std_ = 1.0
            self._fitted = True
            return self
        
        # Remove NaN and specified values
        valid_mask = ~torch.isnan(tensor)
        if self.remove_value is not None:
            valid_mask = valid_mask & (tensor != self.remove_value)
        
        clean_tensor = tensor[valid_mask]
        
        if len(clean_tensor) == 0:
            raise ValueError("No valid data points after removing NaN and specified values")
        
        # Convert to numpy for sklearn fitting
        data_np = clean_tensor.detach().cpu().numpy().reshape(-1, 1)
        self.scale_factor_ = float(1.0 / np.median(data_np))
        data_np = data_np * self.scale_factor_

        
        # Fit sklearn transformer (it handles everything: transformation + standardization)
        self._sklearn_transformer.fit(data_np)
        
        # Store parameters for compatibility with existing code
        self.lambdas_ = torch.tensor(
            self._sklearn_transformer.lambdas_, 
            dtype=torch.float32,
            device=self.device
        )
        
        # When standardize=True, sklearn uses an internal _scaler
        if hasattr(self._sklearn_transformer, '_scaler'):
            self.mean_ = float(self._sklearn_transformer._scaler.mean_[0])
            self.std_ = float(self._sklearn_transformer._scaler.scale_[0])
        else:
            # Fallback for standardize=False
            self.mean_ = 0.0
            self.std_ = 1.0
        
        self._fitted = True
        return self

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor using power transformation.
        
        Args:
            tensor: Input tensor to normalize
            
        Returns:
            Normalized tensor
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before use")
        
        # Store original device and shape
        original_device = tensor.device
        original_shape = tensor.shape
        
        # Convert to numpy
        data_np = tensor.detach().cpu().numpy().reshape(-1, 1)

        # Apply scaling factor before transformation to avoid overlarge lambdas
        data_np = data_np * self.scale_factor_

        # Use sklearn transformer (handles both transformation and standardization)
        try:
            transformed = self._sklearn_transformer.transform(data_np).flatten()
        except Exception as e:
            # Handle potential numerical issues gracefully
            warnings.warn(f"Transformation warning: {e}. Using fallback.")
            transformed = np.zeros_like(data_np.flatten())
        
        # Convert back to tensor
        result = torch.tensor(transformed, dtype=torch.float32, device=original_device)
        
        # Reshape to original shape
        result = result.reshape(original_shape)
        
        return result
    
    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor back to original scale.
        
        Args:
            normed_tensor: Normalized tensor to denormalize
            
        Returns:
            Denormalized tensor in original scale
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before use")
        
        # Store original device and shape
        original_device = normed_tensor.device
        original_shape = normed_tensor.shape
        
        # Convert to numpy
        data_np = normed_tensor.detach().cpu().numpy().reshape(-1, 1)
        
        # For Yeo-Johnson with negative lambda, clamp input to prevent NaN
        # When lambda < 0, the valid range for standardized values is limited
        if self.method == 'yeo-johnson' and self.lambdas_[0] < 0:
            # Calculate the maximum safe value to avoid NaN in inverse transform
            # For negative lambda: base = x*lambda + 1 must be > 0
            # So: x < -1/lambda (in transformed space before standardization)
            lambda_val = self.lambdas_[0].item()
            max_transformed = -1.0 / lambda_val
            
            # Convert to standardized space: (max_transformed - mean) / std
            max_standardized = (max_transformed - self.mean_) / self.std_
            
            # Clamp with a safety margin (99% of the limit)
            safe_max = max_standardized * 0.99
            data_np = np.clip(data_np, -1e6, safe_max)
        
        # Use sklearn inverse transform
        try:
            result_np = self._sklearn_transformer.inverse_transform(data_np).flatten()
            
            # Check for NaN and replace with clamped maximum if needed
            if np.isnan(result_np).any():
                warnings.warn("NaN detected in inverse transform, using safe clamping")
                # Recompute with more aggressive clamping
                if self.lambdas_[0] < 0:
                    lambda_val = self.lambdas_[0].item()
                    max_transformed = -1.0 / lambda_val
                    max_standardized = (max_transformed - self.mean_) / self.std_
                    data_np_safe = np.clip(data_np, -1e6, max_standardized * 0.95)
                    result_np = self._sklearn_transformer.inverse_transform(data_np_safe).flatten()
            
            
                    
        except Exception as e:
            # Handle other potential numerical issues
            warnings.warn(f"Inverse transformation error: {e}. Using fallback.")
            result_np = np.zeros_like(data_np.flatten())
        
        # Reverse scaling factor
        result_np = result_np / self.scale_factor_
        # Convert back to tensor
        result = torch.tensor(result_np, dtype=torch.float32, device=original_device)
        
        # Reshape to original shape
        result = result.reshape(original_shape)
        
        return result
    
    def state_dict(self) -> dict:
        """
        Get state dictionary for saving/loading.
        
        Returns:
            Dictionary containing all necessary parameters
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before getting state dict")
            
        return {
            'name': self.name,
            'method': self.method,
            'remove_value': self.remove_value,
            'lambdas': self.lambdas_.cpu().numpy().tolist() if self.lambdas_ is not None else None,
            'mean': self.mean_,
            'std': self.std_,
            'scale_factor': self.scale_factor_,
            'fitted': self._fitted
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load state dictionary and reconstruct sklearn transformer.
        
        Args:
            state_dict: Dictionary containing parameters
        """
        self.method = state_dict['method']
        self.remove_value = state_dict.get('remove_value', None)
        self.scale_factor_ = state_dict.get('scale_factor', 1.0)  # Default for old models
        
        if state_dict['lambdas'] is not None:
            self.lambdas_ = torch.tensor(
                state_dict['lambdas'], 
                dtype=torch.float32,
                device=self.device
            )
        else:
            self.lambdas_ = None
            
        self.mean_ = state_dict['mean']
        self.std_ = state_dict['std']
        self._fitted = state_dict.get('fitted', False)
        
        # Reconstruct sklearn transformer with loaded parameters
        if self._fitted and self.lambdas_ is not None:
            self._sklearn_transformer = PowerTransformer(
                method=self.method,
                standardize=True,
                copy=self.copy
            )
            
            # Set fitted state
            self._sklearn_transformer.lambdas_ = self.lambdas_.cpu().numpy()
            
            # Manually set the internal StandardScaler state
            from sklearn.preprocessing import StandardScaler
            self._sklearn_transformer._scaler = StandardScaler()
            self._sklearn_transformer._scaler.mean_ = np.array([self.mean_])
            self._sklearn_transformer._scaler.scale_ = np.array([self.std_])
            self._sklearn_transformer._scaler.var_ = np.array([self.std_ ** 2])
            self._sklearn_transformer._scaler.n_features_in_ = 1
            self._sklearn_transformer._scaler.n_samples_seen_ = 1
            
            # Mark sklearn transformer as fitted
            self._sklearn_transformer._fitted = True
    
    def to(self, device: torch.device) -> 'PowerTransformerNormalizer':
        """
        Move normalizer parameters to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        self.device = device
        if self.lambdas_ is not None:
            self.lambdas_ = self.lambdas_.to(device)
        return self

class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, log_labels=False, remove_value=None, normalize=True):
        """Initialize normalizer with tensor statistics."""
        super(Normalizer, self).__init__()
        
        self.name = 'standard'
        self.log_labels = log_labels
        self.remove_value = remove_value
        self.device = torch.device('cpu')
        self.normalize = normalize

    def fit(self, tensor):
        """Fit normalizer to tensor."""
        # Remove NaN and specified values for normalization
        tensor = tensor[torch.isnan(tensor) == False]
        if self.remove_value is not None:
            tensor = tensor[tensor != self.remove_value]
            
        if hasattr(self, 'log_labels') and self.log_labels:
            tensor = torch.log10(tensor + 1e-6)  # avoid log10(0)
            print("Log10(x+1) transform applied to labels.")
        if self.normalize:
            self.mean = torch.mean(tensor, dim=0)
            self.std = torch.std(tensor, dim=0)
        else:
            self.mean = torch.tensor(0.0)
            self.std = torch.tensor(1.0)
        self.mean_ = float(self.mean.cpu().numpy())
        self.std_ = float(self.std.cpu().numpy())
        self.device = tensor.device

    def norm(self, tensor):
        """Normalize tensor."""
        if hasattr(self, 'log_labels') and self.log_labels:
            tensor = torch.log10(tensor + 1e-6)
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        """Denormalize tensor."""
        denormed_tensor = normed_tensor * self.std + self.mean
        if hasattr(self, 'log_labels') and self.log_labels:
            denormed_tensor = torch.clamp(denormed_tensor, -20, 20)  # avoid numerical errors
            return torch.pow(10, denormed_tensor) - 1e-6
        else:
            return denormed_tensor

    def state_dict(self):
        """Get state dictionary."""
        return {
            'name': self.name,
            'mean': self.mean_, 
            'std': self.std_,
            'log_labels': self.log_labels,
            'remove_value': self.remove_value,
            'normalize': self.normalize
                }

    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        self.mean_ = state_dict['mean']
        self.std_ = state_dict['std']
        self.log_labels = state_dict.get('log_labels', False)
        self.remove_value = state_dict.get('remove_value', None)
        self.normalize = state_dict.get('normalize', True)
        self.mean = torch.tensor(self.mean_).to(self.device)
        self.std = torch.tensor(self.std_).to(self.device)
        
    def to(self, device):
        """Move normalizer to device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.device = device
        return self