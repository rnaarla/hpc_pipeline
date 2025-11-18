"""Unit tests for helper functions."""
import pytest
import torch

@pytest.mark.unit
class TestHelperFunctions:
    """Test utility and helper functions."""
    
    def test_device_selection(self, device):
        """Test device selection logic."""
        assert device.type in ["cuda", "cpu"]
        
    def test_tensor_creation(self, device):
        """Test tensor creation on correct device."""
        tensor = torch.randn(10, 10, device=device)
        assert tensor.device.type == device.type
        
    def test_config_validation(self, sample_config):
        """Test configuration validation."""
        assert "batch_size" in sample_config
        assert sample_config["batch_size"] > 0
        assert sample_config["learning_rate"] > 0
