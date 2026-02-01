"""
Sintellix - Neural Network Framework with HOT Architecture

A PyTorch-like framework for building and training neural networks
with Higher Order Thought (HOT) architecture.
"""

__version__ = "0.1.0"

# Import core components
from .core import (
    NeuronModel,
    NeuronConfig,
    Neuron,
)

# Import training utilities
from .training import (
    Trainer,
    TrainingConfig,
)

# Import model management
from .models import (
    ModelManager,
    download_model,
)

# Import channel operations (critical for saltts project)
from .channel_ops import (
    SubchannelExtractor,
    ChannelWrapper,
    AuxiliaryChannelInserter,
    ChannelType,
    extract_and_wrap_text,
    extract_and_wrap_audio,
    extract_wrap_and_insert,
)

__all__ = [
    # Core
    "NeuronModel",
    "NeuronConfig",
    "Neuron",
    # Training
    "Trainer",
    "TrainingConfig",
    # Models
    "ModelManager",
    "download_model",
    # Channel Operations (saltts integration)
    "SubchannelExtractor",
    "ChannelWrapper",
    "AuxiliaryChannelInserter",
    "ChannelType",
    "extract_and_wrap_text",
    "extract_and_wrap_audio",
    "extract_wrap_and_insert",
]
