import unittest
from data_loader import create_data_loaders

class TestDataLoader(unittest.TestCase):

    def test_data_loaders_creation(self):
        """Test that data loaders are created without errors and have correct properties."""

        data_dir = './tests/'  # Ensure this points to a directory with test data
        loaders = create_data_loaders(data_dir=data_dir, batch_size=2, num_workers=0)
        
        # Check that all loaders are created
        self.assertIn('train', loaders)
        self.assertIn('val', loaders)
        self.assertIn('unlabelled', loaders)
        
        
        # Additional tests can be added here to check for specific transformations, data ranges, etc.

if __name__ == '__main__':
    unittest.main()