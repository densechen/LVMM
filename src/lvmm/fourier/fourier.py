import torch
import torch.nn as nn


class Fourier(nn.Module):
    def __init__(self, k_frequency, num_frames):
        """
        Args:
            k: keep top k Fourier Frequency Coefficients.
        """
        super().__init__()
        
        self.k_frequency = k_frequency
        self.num_frames = num_frames
        
    def forward(self, x):
        """
        Args:
            x: batch size, channels, frames, height, width
        """
        y = torch.fft.rfft(x, dim=2)
        
        if self.k_frequency is not None:
            return y[:, :, :self.k_frequency]
        else:
            return y

    def reverse(self, y, n):
        x_inv = torch.fft.irfft(y, n, dim=2)

        return x_inv

    @staticmethod
    def complex_to_real_and_imag(x):
        """
        Args:
            x: batch size, channels, k, height, width
        """
        return torch.cat([x.real, x.imag], dim=1)
    
    @staticmethod
    def real_and_imag_to_complex(x):
        """
        Args:
            x: batch size, channels * 2, k, height, width
        """
        return torch.complex(*x.chunk(chunks=2, dim=1))
    
    def reconstruct(self, x):
        """
        Args:
            x: batch size, channels * 2, k, height, width
        Returns:
            x: batch size, channels, num frames, height, width
        """
        return self.reverse(self.real_and_imag_to_complex(x), n=self.num_frames)
    
    def deconstruct(self, x):
        """
        Args:
            x: batch size, channel, num frames, height, width
        Returns:
            x: batch size, channels * 2, k, height, width
        """
        return self.complex_to_real_and_imag(self(x))
    
    def loss(self, x):
        """Compute the estimation loss of Fourier transform.
        Args:
            x: batch size, channels, frames, height, width
        """
        y = self.deconstruct(x)
        inv_x = self.reconstruct(y)
        return torch.norm(x-inv_x)
    
if __name__ == "__main__":
    fourier = Fourier(k=64)
    
    x = torch.randn(4, 3, 128, 64, 64)
    
    y = fourier(x)
    
    z = fourier.complex_to_real_and_imag(y)
    inv_z = fourier.real_and_imag_to_complex(z)
    
    print(torch.abs(inv_z - y).sum())
    
    inv_x = fourier.reverse(y, n=128)
    
    print(torch.norm(x - inv_x))
    
    print(fourier.loss(x))