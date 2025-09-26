"""
Power Transformer Normalizer
Advanced data transformer based on sklearn PowerTransformer with PyTorch compatibility
Author: zhangshd
Date: September 26, 2025
"""

import torch
import numpy as np
from sklearn.preprocessing import PowerTransformer
from typing import Optional, Union, Literal
import warnings


class PowerTransformerNormalizer(object):
    """
    Advanced normalizer using Power Transformer with PyTorch GPU support.
    
    Combines sklearn's PowerTransformer for fitting with pure PyTorch implementation 
    for GPU-accelerated transformation during training and inference.
    
    Supports both Box-Cox and Yeo-Johnson transformations:
    - Box-Cox: Only for strictly positive data
    - Yeo-Johnson: Works with any real values (including negative and zero)
    
    Maintains full compatibility with the original Normalizer interface.
    """

    def __init__(
        self,
        method: Literal['box-cox', 'yeo-johnson'] = 'yeo-johnson',
        log_labels: bool = False,
        remove_value: Optional[float] = None,
        copy: bool = True
    ):
        """
        Initialize Power Transformer Normalizer.
        
        Args:
            method: Transformation method ('box-cox' or 'yeo-johnson')
            log_labels: Whether to apply log10 transformation (for backward compatibility)
            remove_value: Value to remove from data before fitting
            copy: Whether to copy data during transformation
        """
        super(PowerTransformerNormalizer, self).__init__()
        
        self.method = method
        self.log_labels = log_labels
        self.remove_value = remove_value
        self.copy = copy
        self.device = torch.device('cpu')
        
        # sklearn transformer for fitting (always disable standardize to handle it manually)
        self._sklearn_transformer = PowerTransformer(
            method=method,
            standardize=False,
            copy=copy
        )
        
        # Parameters for PyTorch implementation
        self.lambdas_ = None
        self.mean_ = None
        self.std_ = None
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
        
        # Handle log transformation first (for backward compatibility)
        if self.log_labels:
            data_np = np.log10(data_np + 1e-5)
            print("Log10(x+1e-5) transform applied to labels.")
        
        # Fit sklearn transformer
        self._sklearn_transformer.fit(data_np)
        
        # Extract parameters for PyTorch implementation
        self.lambdas_ = torch.tensor(
            self._sklearn_transformer.lambdas_, 
            dtype=torch.float32,
            device=self.device
        )
        
        # Apply power transformation to get transformed data for standardization
        transformed_data = self._sklearn_transformer.transform(data_np)
        transformed_tensor = torch.tensor(transformed_data.flatten(), dtype=torch.float32, device=self.device)
        
        # Calculate mean and std for standardization using PyTorch
        self.mean_ = float(torch.mean(transformed_tensor).cpu().numpy())
        self.std_ = float(torch.std(transformed_tensor).cpu().numpy())
            
        self._fitted = True
        return self
    
    def _power_transform_pytorch(self, x: torch.Tensor, lmbda: torch.Tensor) -> torch.Tensor:
        """
        Apply power transformation using pure PyTorch operations.
        
        Args:
            x: Input tensor
            lmbda: Lambda parameter for transformation
            
        Returns:
            Transformed tensor
        """
        if self.method == 'yeo-johnson':
            return self._yeo_johnson_transform(x, lmbda)
        elif self.method == 'box-cox':
            return self._box_cox_transform(x, lmbda)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _yeo_johnson_transform(self, x: torch.Tensor, lmbda: torch.Tensor) -> torch.Tensor:
        """Yeo-Johnson transformation in PyTorch."""
        eps = 1e-8
        lmbda = lmbda + eps  # Avoid division by zero
        
        # Case 1: x >= 0 and lambda != 0
        mask1 = (x >= 0) & (torch.abs(lmbda) > eps)
        # Case 2: x >= 0 and lambda == 0  
        mask2 = (x >= 0) & (torch.abs(lmbda) <= eps)
        # Case 3: x < 0 and lambda != 2
        mask3 = (x < 0) & (torch.abs(lmbda - 2) > eps)
        # Case 4: x < 0 and lambda == 2
        mask4 = (x < 0) & (torch.abs(lmbda - 2) <= eps)
        
        result = torch.zeros_like(x)
        
        if mask1.any():
            result[mask1] = (torch.pow(x[mask1] + 1, lmbda) - 1) / lmbda
        
        if mask2.any():
            result[mask2] = torch.log(x[mask2] + 1)
            
        if mask3.any():
            result[mask3] = -(torch.pow(-x[mask3] + 1, 2 - lmbda) - 1) / (2 - lmbda)
            
        if mask4.any():
            result[mask4] = -torch.log(-x[mask4] + 1)
            
        return result
    
    def _box_cox_transform(self, x: torch.Tensor, lmbda: torch.Tensor) -> torch.Tensor:
        """Box-Cox transformation in PyTorch."""
        if torch.any(x <= 0):
            raise ValueError("Box-Cox transformation requires strictly positive data")
        
        eps = 1e-8
        
        if torch.abs(lmbda) > eps:
            return (torch.pow(x, lmbda) - 1) / lmbda
        else:
            return torch.log(x)
    
    def _inverse_power_transform_pytorch(self, x: torch.Tensor, lmbda: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse power transformation using pure PyTorch operations.
        
        Args:
            x: Transformed tensor
            lmbda: Lambda parameter for transformation
            
        Returns:
            Original scale tensor
        """
        if self.method == 'yeo-johnson':
            return self._yeo_johnson_inverse_transform(x, lmbda)
        elif self.method == 'box-cox':
            return self._box_cox_inverse_transform(x, lmbda)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _yeo_johnson_inverse_transform(self, x: torch.Tensor, lmbda: torch.Tensor) -> torch.Tensor:
        """Inverse Yeo-Johnson transformation in PyTorch."""
        eps = 1e-8
        lmbda = lmbda + eps
        
        # Case 1: x >= 0 and lambda != 0
        mask1 = (x >= 0) & (torch.abs(lmbda) > eps)
        # Case 2: x >= 0 and lambda == 0
        mask2 = (x >= 0) & (torch.abs(lmbda) <= eps)
        # Case 3: x < 0 and lambda != 2
        mask3 = (x < 0) & (torch.abs(lmbda - 2) > eps)
        # Case 4: x < 0 and lambda == 2
        mask4 = (x < 0) & (torch.abs(lmbda - 2) <= eps)
        
        result = torch.zeros_like(x)
        
        if mask1.any():
            result[mask1] = torch.pow(x[mask1] * lmbda + 1, 1 / lmbda) - 1
            
        if mask2.any():
            result[mask2] = torch.exp(x[mask2]) - 1
            
        if mask3.any():
            result[mask3] = -(torch.pow(-x[mask3] * (2 - lmbda) + 1, 1 / (2 - lmbda)) - 1)
            
        if mask4.any():
            result[mask4] = -torch.exp(-x[mask4]) + 1
            
        return result
    
    def _box_cox_inverse_transform(self, x: torch.Tensor, lmbda: torch.Tensor) -> torch.Tensor:
        """Inverse Box-Cox transformation in PyTorch."""
        eps = 1e-8
        
        if torch.abs(lmbda) > eps:
            result = torch.pow(x * lmbda + 1, 1 / lmbda)
        else:
            result = torch.exp(x)
            
        return result
    
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
        
        # Ensure tensor is on the same device as normalizer
        original_device = tensor.device
        tensor = tensor.to(self.device)
        
        # Handle log transformation first (for backward compatibility)
        if self.log_labels:
            tensor = torch.log10(tensor + 1e-5)
        
        # Apply power transformation
        transformed = self._power_transform_pytorch(tensor, self.lambdas_[0])
        
        # Apply standardization
        mean_tensor = torch.tensor(self.mean_, device=self.device)
        std_tensor = torch.tensor(self.std_, device=self.device)
        transformed = (transformed - mean_tensor) / std_tensor
            
        return transformed.to(original_device)
    
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
        
        # Ensure tensor is on the same device as normalizer
        original_device = normed_tensor.device
        normed_tensor = normed_tensor.to(self.device)
        
        # Reverse standardization
        mean_tensor = torch.tensor(self.mean_, device=self.device)
        std_tensor = torch.tensor(self.std_, device=self.device)
        denormed = normed_tensor * std_tensor + mean_tensor
        
        # Apply inverse power transformation
        result = self._inverse_power_transform_pytorch(denormed, self.lambdas_[0])
        
        # Reverse log transformation if applied (for backward compatibility)
        if self.log_labels:
            result = torch.clamp(result, -20, 20)  # avoid numerical errors
            result = torch.pow(10, result) - 1e-5
            
        return result.to(original_device)
    
    def state_dict(self) -> dict:
        """
        Get state dictionary for saving/loading.
        
        Returns:
            Dictionary containing all necessary parameters
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before getting state dict")
            
        return {
            'method': self.method,
            'log_labels': self.log_labels,
            'remove_value': self.remove_value,
            'lambdas': self.lambdas_.cpu().numpy().tolist() if self.lambdas_ is not None else None,
            'mean': self.mean_,
            'std': self.std_,
            'fitted': self._fitted
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load state dictionary.
        
        Args:
            state_dict: Dictionary containing parameters
        """
        self.method = state_dict['method']
        self.log_labels = state_dict.get('log_labels', False)
        self.remove_value = state_dict.get('remove_value', None)
        
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


# For backward compatibility, create an alias
Normalizer = PowerTransformerNormalizer