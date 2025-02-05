import torch
from abc import ABC, abstractmethod

class BaseTask(ABC):
    """Base class for all tasks in GFlowNet.
    
    This abstract class defines the interface that all tasks must implement
    for use with the GFlowNet training pipeline.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def energy(self, x):
        """Compute energy for given coordinates.
        
        Args:
            x (torch.Tensor): Input coordinates
            
        Returns:
            torch.Tensor: Energy values
        """
        pass

    @abstractmethod
    def score(self, x):
        """Compute score (gradient of negative energy) for given coordinates.
        
        Args:
            x (torch.Tensor): Input coordinates
            
        Returns:
            torch.Tensor: Score values (forces)
        """
        pass

    @abstractmethod
    def log_reward(self, x):
        """Compute log reward for given coordinates.
        
        Args:
            x (torch.Tensor): Input coordinates
            
        Returns:
            torch.Tensor: Log reward values
        """
        pass

    def get_state(self):
        """Get current state information.
        
        Returns:
            tuple: Current positions and forces
        """
        return None, None

    def setup_openmm_system(self):
        """Setup OpenMM system if needed.
        
        This is optional for tasks that don't use OpenMM.
        """
        pass 