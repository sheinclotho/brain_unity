"""Unit tests for configuration"""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unity_integration.config import TwinBrainConfig

class TestConfig(unittest.TestCase):
    def test_default_config(self):
        config = TwinBrainConfig()
        self.assertEqual(config.model.n_regions, 200)

if __name__ == '__main__':
    unittest.main()
