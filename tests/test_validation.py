"""Unit tests for validation utilities"""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from unity_integration.validation import validate_region_ids, validate_amplitude, ValidationError

class TestValidation(unittest.TestCase):
    def test_valid_region_ids(self):
        result = validate_region_ids([1, 5, 10], n_regions=200)
        self.assertEqual(result, [1, 5, 10])
    
    def test_valid_amplitude(self):
        result = validate_amplitude(0.5)
        self.assertEqual(result, 0.5)

if __name__ == '__main__':
    unittest.main()
