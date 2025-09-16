"""
CGCNN Package
Crystal Graph Convolutional Neural Networks for MOF property prediction.
Aligned with moftransformer structure for consistency.
"""

from . import datamodule
from . import module

__version__ = "2.0.0"
__all__ = ['datamodule', 'module']