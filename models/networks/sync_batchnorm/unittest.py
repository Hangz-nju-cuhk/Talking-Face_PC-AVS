import unittest
import torch


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, x, y):
        adiff = float((x - y).abs().max())
        if (y == 0).all():
            rdiff = 'NaN'
        else:
            rdiff = float((adiff / y).abs().max())

        message = (
            'Tensor close check failed\n'
            'adiff={}\n'
            'rdiff={}\n'
        ).format(adiff, rdiff)
        self.assertTrue(torch.allclose(x, y), message)

