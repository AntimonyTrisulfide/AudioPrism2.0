import unittest
from pathlib import Path

import torch

from audioprism.data import PreprocessedComplexDataset, polar_to_channels


class DatasetTest(unittest.TestCase):
    def test_polar_conversion(self):
        magnitude = torch.tensor([[1.0, 2.0]])
        phase = torch.tensor([[0.0, torch.pi / 2]])
        channels = polar_to_channels(magnitude, phase)
        torch.testing.assert_close(channels[0], torch.tensor([[1.0, 0.0]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(channels[1], torch.tensor([[0.0, 2.0]]), atol=1e-6, rtol=0)

    def test_missing_metadata_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing dataset metadata"):
            PreprocessedComplexDataset(Path("definitely_missing_dataset"))


if __name__ == "__main__":
    unittest.main()
