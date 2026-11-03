import torch


def fft_high_pass_filter(x, cutoff_ratio=0.1):
    fft = torch.fft.fft2(x, norm="ortho")
    fft_shifted = torch.fft.fftshift(fft)

    B, C, H, W = x.shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                          torch.linspace(-1, 1, W, device=x.device))
    radius = torch.sqrt(x**2 + y**2)
    high_pass_mask = (radius > cutoff_ratio).float()

    fft_high = fft_shifted * high_pass_mask

    fft_high = torch.fft.ifftshift(fft_high)
    x_high = torch.fft.ifft2(fft_high, norm="ortho").real

    return x_high
