from numpy import abs, arange, empty, real, fft, sin, cos
from cmath import exp, pi


def generate_ramp_array(n):
    results = empty(n, float)
    half = (n - 1) // 2 + 1
    left = arange(0, half, dtype=int)
    results[:half] = left
    right = arange(-(n // 2), 0, dtype=int)
    results[half:] = right
    return results / n


def DFT(F):
    M, N = F.shape
    range_N = range(N)
    range_M = range(M)
    exp_item = -1j * pi * 2
    return [
        [sum([sum([F[m, n] * exp(exp_item * ((k / M) * m + (l / N) * n)) for n in range_N]) for m in range_M]) for l in
         range_N] for k in range_M]


def IDFT(F):
    M, N = F.shape
    range_N = range(N)
    range_M = range(M)
    exp_item = 1j * pi * 2
    return [
        [sum([sum([F[m, n] * exp(exp_item * ((k / M) * m + (l / N) * n)) for n in range_N]) for m in range_M]) / M * N
         for l in range_N] for k in range_M]


def filter_sinogram(sinogram, type):
    detector_quantity = sinogram.shape[1]
    ramp_array = generate_ramp_array(detector_quantity)
    omega = 2 * pi * ramp_array
    fourier_filter = 2 * abs(ramp_array)
    if type == "Shepp-logan":
        fourier_filter[1:] = fourier_filter[1:] * sin(omega[1:]) / omega[1:]
    elif type == "Cosine":
        fourier_filter *= cos(omega)
    elif type == "Hamming":
        fourier_filter *= (0.54 + 0.46 * cos(omega / 2))
    elif type == "Hann":
        fourier_filter *= (1 + cos(omega / 2)) / 2
    # sinogram_freq_domain_filtered = DFT(sinogram) * ramp_filter
    # sinogram_filtered = real(IDFT(sinogram_freq_domain_filtered))
    sinogram_freq_domain_filtered = fft.fft(sinogram) * fourier_filter  # numpy fft is better optimized and much quicker
    sinogram_filtered = real(fft.ifft(sinogram_freq_domain_filtered))
    return sinogram_filtered
