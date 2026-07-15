import unittest

import torch

from audioprism.config import ModelConfig
from audioprism.model import AudioPrism, mel_band_edges


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = ModelConfig(num_bands=4, dim=16, depth=1, heads=4, ff_mult=2, dropout=0.0)
        self.model = AudioPrism(n_freqs=17, num_sources=3, sample_rate=16000, n_fft=32, config=config)

    def test_forward_shapes_and_finite_values(self):
        mixture = torch.randn(2, 2, 17, 9)
        output = self.model(mixture)
        self.assertEqual(output["estimates"].shape, (2, 3, 2, 17, 9))
        self.assertEqual(output["mask"].shape, (2, 3, 2, 17, 9))
        self.assertEqual(output["activity_logits"].shape, (2, 3))
        self.assertTrue(torch.isfinite(output["estimates"]).all())

    def test_mixture_consistency(self):
        mixture = torch.randn(2, 2, 17, 9)
        estimates = self.model(mixture)["estimates"]
        torch.testing.assert_close(estimates.sum(dim=1), mixture, atol=2e-5, rtol=2e-5)

    def test_band_edges_cover_frequency_axis(self):
        edges = mel_band_edges(1025, 32, 16000, 2048)
        self.assertEqual(edges[0], 0)
        self.assertEqual(edges[-1], 1025)
        self.assertTrue(all(right > left for left, right in zip(edges, edges[1:])))


if __name__ == "__main__":
    unittest.main()
