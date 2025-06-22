"""
GPU ускоренный FFT процессор
"""

import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import time


class GPUFFTProcessor:
    """GPU ускоренный FFT процессор"""
    
    def __init__(self, acceleration_type="OpenCL"):
        self.acceleration_type = acceleration_type
        self.use_gpu = acceleration_type != "CPU"
        
    def fft2_gpu(self, image):
        """GPU ускоренный FFT2"""
        if not self.use_gpu:
            return fft2(image)
            
        if self.acceleration_type == "CUDA":
            return self._fft2_cuda(image)
        elif self.acceleration_type == "OpenCL":
            return self._fft2_opencl(image)
        else:
            return fft2(image)
    
    def ifft2_gpu(self, spectrum):
        """GPU ускоренный IFFT2"""
        if not self.use_gpu:
            return ifft2(spectrum)
            
        if self.acceleration_type == "CUDA":
            return self._ifft2_cuda(spectrum)
        elif self.acceleration_type == "OpenCL":
            return self._ifft2_opencl(spectrum)
        else:
            return ifft2(spectrum)
    
    def _fft2_cuda(self, image):
        """CUDA FFT2"""
        try:
            # Загружаем на GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image.astype(np.float32))
            
            # CUDA FFT
            gpu_fft = cv2.cuda_GpuMat()
            cv2.cuda.dft(gpu_image, gpu_fft, gpu_image.size())
            
            # Скачиваем результат
            result = gpu_fft.download()
            
            # Преобразуем в стандартный формат numpy комплексных чисел
            if result.ndim == 3 and result.shape[2] == 2:
                complex_result = result[:, :, 0] + 1j * result[:, :, 1]
                return complex_result
            else:
                return result
        except Exception as e:
            print(f"CUDA FFT failed: {e}, falling back to CPU")
            return fft2(image)
    
    def _ifft2_cuda(self, spectrum):
        """CUDA IFFT2"""
        try:
            # Преобразуем комплексные числа в формат CUDA
            if spectrum.dtype == np.complex128 or spectrum.dtype == np.complex64:
                real_part = np.real(spectrum).astype(np.float32)
                imag_part = np.imag(spectrum).astype(np.float32)
                cuda_spectrum = np.stack([real_part, imag_part], axis=2)
            else:
                cuda_spectrum = spectrum.astype(np.float32)
            
            # Загружаем на GPU
            gpu_spectrum = cv2.cuda_GpuMat()
            gpu_spectrum.upload(cuda_spectrum)
            
            # CUDA IFFT
            gpu_ifft = cv2.cuda_GpuMat()
            cv2.cuda.idft(gpu_spectrum, gpu_ifft, gpu_spectrum.size())
            
            # Скачиваем результат
            result = gpu_ifft.download()
            return result
        except Exception as e:
            print(f"CUDA IFFT failed: {e}, falling back to CPU")
            return ifft2(spectrum)
    
    def _fft2_opencl(self, image):
        """OpenCL FFT2"""
        try:
            # Включаем OpenCL
            if not cv2.ocl.useOpenCL():
                return fft2(image)
            
            # OpenCV автоматически использует OpenCL для DFT
            # Возвращаем в стандартном формате комплексных чисел
            result = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
            # Преобразуем в стандартный формат numpy комплексных чисел
            complex_result = result[:, :, 0] + 1j * result[:, :, 1]
            return complex_result
        except Exception as e:
            print(f"OpenCL FFT failed: {e}, falling back to CPU")
            return fft2(image)
    
    def _ifft2_opencl(self, spectrum):
        """OpenCL IFFT2"""
        try:
            # Включаем OpenCL
            if not cv2.ocl.useOpenCL():
                return ifft2(spectrum)
            
            # Преобразуем комплексные числа в формат OpenCV
            if spectrum.dtype == np.complex128 or spectrum.dtype == np.complex64:
                real_part = np.real(spectrum).astype(np.float32)
                imag_part = np.imag(spectrum).astype(np.float32)
                opencv_spectrum = np.stack([real_part, imag_part], axis=2)
            else:
                opencv_spectrum = spectrum.astype(np.float32)
            
            # OpenCV автоматически использует OpenCL для IDFT
            result = cv2.idft(opencv_spectrum, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            return result
        except Exception as e:
            print(f"OpenCL IFFT failed: {e}, falling back to CPU")
            return ifft2(spectrum) 