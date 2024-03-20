import unittest
import torch
from train import entropy_volume  

class TestEntropyVolume(unittest.TestCase):

    def test_entropy_computation(self):
        """
        Test the entropy computation for a known input and compare it against the expected output.
        """
        # Create a test input tensor with shape [MC samples, Channels, Spatial dimensions]
        # For simplicity, using a small tensor and manually computing expected result
        # Test case setup: 3 MC samples, 1 channel, spatial dimensions of 2x2x2
        vol_input = torch.tensor([
            [[[
                [0.8, 0.2], 
                [0.1, 0.9]
            ], [
                [0.5, 0.5], 
                [0.7, 0.3]
            ]]],
            
            [[[
                [0.6, 0.4], 
                [0.2, 0.8]
            ], [
                [0.4, 0.6], 
                [0.6, 0.4]
            ]]],
            
            [[[
                [0.9, 0.1], 
                [0.3, 0.7]
            ], [
                [0.55, 0.45], 
                [0.65, 0.35]
            ]]]
        ], dtype=torch.float32)

        # Expected entropy computed manually for this simple case
        # The input tensor represents 3 MC samples with 1 channels (classes) and  spatial dimensions (2x2x2)
        # Expected entropy values are calculated for each spatial location
        expected_entropy = torch.tensor([[[0.2037, 0.3396],[0.3219, 0.1785]],   
                                         [[0.3514, 0.3412],[0.2800, 0.3674]]])

        # Calculate the entropy using the function
        computed_entropy = entropy_volume(vol_input)

        # Check if the computed entropy is close to the expected values
        self.assertTrue(torch.allclose(computed_entropy, expected_entropy, atol=1e-4),
                        msg=f"Computed entropy {computed_entropy} does not match expected {expected_entropy}")

if __name__ == "__main__":
    unittest.main()