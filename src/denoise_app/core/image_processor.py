import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import gc

from .wavelet_processor import WaveletProcessor

try:
    from .gpu_fft import GPUFFTProcessor
    from .gpu_fft import ACCELERATION_TYPE
except ImportError:
    ACCELERATION_TYPE = "CPU"

    class GPUFFTProcessor:
        def __init__(self, acc_type="CPU"):
            pass

        def fft2_gpu(self, img):
            return fft2(img)

        def ifft2_gpu(self, spec):
            return ifft2(spec)


class NoiseGenerator:
    def __init__(self):
        self.freq = 0.1
        self.amplitude = 0.3
        self.angle = 45
        self.noise_type = "periodic"
        self.gaussian_mean = 0.0
        self.gaussian_std = 0.05

    def set_noise_type(self, noise_type: str):
        if noise_type in ["periodic", "gaussian"]:
            self.noise_type = noise_type

    def _ensure_float(self, image):
        if image is None:
            return None
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        if image.dtype == np.float64:
            return image.astype(np.float32)
        if image.dtype == np.float32:
            return image

        return image.astype(np.float32)

    def add_noise(self, image):
        image = self._ensure_float(image)
        if image is None:
            return None

        if self.noise_type == "periodic":
            return self._add_periodic_noise(image)
        elif self.noise_type == "gaussian":
            return self._add_gaussian_noise(image)
        return image

    def _add_periodic_noise(self, image):
        height, width = image.shape[:2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        rad_angle = np.radians(self.angle)
        freq_x = self.freq * np.cos(rad_angle)
        freq_y = self.freq * np.sin(rad_angle)

        noise = (self.amplitude * np.sin(2 * np.pi * (freq_x * x + freq_y * y))).astype(
            np.float32
        )

        if len(image.shape) == 3:
            noise = np.expand_dims(noise, axis=2)

        noisy_image = image + noise
        return np.clip(noisy_image, 0.0, 1.0)

    def _add_gaussian_noise(self, image):
        noise = np.random.normal(
            self.gaussian_mean, self.gaussian_std, image.shape
        ).astype(np.float32)
        return np.clip(image + noise, 0.0, 1.0)


class FrequencyFilter:
    def __init__(self):
        self.filter_type = "Butterworth"
        self.D0 = 0.2
        self.n = 2
        self.padding_ratio = 0.1
        self.blend_alpha = 1.0
        self.enable_blend = False

        self.gpu_fft = GPUFFTProcessor(ACCELERATION_TYPE)

        self._cached_mask = None
        self._cache_key = None

    def _get_mask(self, height, width):
        current_key = (height, width, self.filter_type, self.D0, self.n)

        if self._cached_mask is not None and self._cache_key == current_key:
            return self._cached_mask

        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2

        dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
        max_dist_sq = center_x**2 + center_y**2

        with np.errstate(divide="ignore", invalid="ignore"):
            if self.filter_type == "Ideal":
                distances_sq = dist_sq / max_dist_sq
                mask = (distances_sq <= self.D0**2).astype(np.float32)

            elif self.filter_type == "Gaussian":
                distances_sq = dist_sq / max_dist_sq
                mask = np.exp(-distances_sq / (2 * self.D0**2))

            elif self.filter_type == "Butterworth":
                distances = np.sqrt(dist_sq / max_dist_sq)
                mask = 1 / (1 + (distances / self.D0) ** (2 * self.n))
                mask[np.isnan(mask)] = 0
            else:
                mask = np.ones((height, width), dtype=np.float32)

        self._cached_mask = mask.astype(np.float32)
        self._cache_key = current_key
        return self._cached_mask

    def apply_filter(self, image):
        if image is None:
            return None

        if image.dtype != np.float32:
            image = image.astype(np.float32)

        pad_h = int(image.shape[0] * self.padding_ratio)
        pad_w = int(image.shape[1] * self.padding_ratio)

        if len(image.shape) == 3:
            padded = cv2.copyMakeBorder(
                image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT
            )
        else:
            padded = cv2.copyMakeBorder(
                image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT
            )

        p_h, p_w = padded.shape[:2]
        mask = self._get_mask(p_h, p_w)

        result = np.zeros_like(padded)

        if len(padded.shape) == 3:
            for i in range(3):
                f_transform = self.gpu_fft.fft2_gpu(padded[:, :, i])
                f_shift = fftshift(f_transform)
                filtered_shift = f_shift * mask
                f_ishift = ifftshift(filtered_shift)
                result[:, :, i] = np.real(self.gpu_fft.ifft2_gpu(f_ishift))
        else:
            f_transform = self.gpu_fft.fft2_gpu(padded)
            f_shift = fftshift(f_transform)
            filtered_shift = f_shift * mask
            f_ishift = ifftshift(filtered_shift)
            result = np.real(self.gpu_fft.ifft2_gpu(f_ishift))

        if pad_h > 0 and pad_w > 0:
            result = result[pad_h:-pad_h, pad_w:-pad_w]

        result = np.clip(result, 0.0, 1.0)

        if self.enable_blend and self.blend_alpha < 1.0:
            return result * self.blend_alpha + image * (1.0 - self.blend_alpha)

        return result

    def set_params(self, type_name, d0, n):
        self.filter_type = type_name
        self.D0 = max(0.001, d0)
        self.n = n


class DetailEnhancer:
    def __init__(self):
        self.sigma_s = 1.0
        self.sigma_r = 0.5
        self.contrast = 0.0

    def enhance_details(self, image):
        if image is None:
            return None
        if self.sigma_r <= 0 and self.contrast <= 0:
            return image

        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if self.sigma_r > 0:
            blur_ksize = int(self.sigma_s * 6) | 1
            blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), self.sigma_s)
            sharpened = cv2.addWeighted(
                image, 1.0 + self.sigma_r, blurred, -self.sigma_r, 0
            )
        else:
            sharpened = image

        if self.contrast > 0:
            temp_uint = (np.clip(sharpened, 0, 1) * 255).astype(np.uint8)
            lab = cv2.cvtColor(temp_uint, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=1.0 + (self.contrast * 2.0), tileGridSize=(8, 8)
            )
            l = clahe.apply(l)

            lab_merged = cv2.merge((l, a, b))
            result_uint = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)
            return result_uint.astype(np.float32) / 255.0

        return np.clip(sharpened, 0.0, 1.0)

    def set_params(self, sigma_s, sigma_r, contrast):
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.contrast = contrast


class ImageProcessor:
    def __init__(self):
        self.noise_gen = NoiseGenerator()
        self.filter = FrequencyFilter()
        self.enhancer = DetailEnhancer()
        self.wavelet = WaveletProcessor()
        self.use_gpu = ACCELERATION_TYPE != "CPU"

    def process_image_wavelet(self, original):
        img = self.noise_gen._ensure_float(original)
        noisy = self.noise_gen.add_noise(img)
        cleaned = self.wavelet.process_image(noisy)
        mosaic = self.wavelet.generate_mosaic(noisy)

        return noisy, cleaned, (None, None, mosaic)

    def set_wavelet_parameters(
        self,
        wavelet,
        level,
        threshold,
        mode,
        post_smooth=False,
        use_swt=False,
        chroma_str=0.5,
    ):
        self.wavelet.set_params(
            wavelet,
            level,
            threshold,
            mode,
            post_smooth=post_smooth,
            use_swt=use_swt,
            chroma_str=chroma_str,
        )

    def process_image(self, original):
        img = self.noise_gen._ensure_float(original)
        noisy = self.noise_gen.add_noise(img)
        filtered = self.filter.apply_filter(noisy)
        final = self.enhancer.enhance_details(filtered)
        spectra = self._compute_spectra_set(img, noisy, final)
        return noisy, final, spectra

    def apply_frequency_filter(self, image):
        return self.filter.apply_filter(image)

    def enhance_details(self, image):
        return self.enhancer.enhance_details(image)

    def force_memory_cleanup(self):
        gc.collect()

    def compute_spectrum(self, image):
        if image is None:
            return None

        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-9)

        mn, mx = magnitude.min(), magnitude.max()
        if mx - mn > 0:
            magnitude = (magnitude - mn) / (mx - mn)
        else:
            magnitude = np.zeros_like(magnitude)

        return magnitude

    def _compute_spectra_set(self, orig, noisy, filtered):
        s1 = self.compute_spectrum(orig)
        s2 = self.compute_spectrum(noisy)
        s3 = self.compute_spectrum(filtered)
        return (s1, s2, s3)

    def set_filter_parameters(self, f_type, d0, n):
        self.filter.set_params(f_type, d0, n)

    def set_noise_parameters(self, freq, amp, angle):
        self.noise_gen.freq = freq
        self.noise_gen.amplitude = amp
        self.noise_gen.angle = angle

    def set_enhancement_parameters(self, s_s, s_r, contrast):
        self.enhancer.set_params(s_s, s_r, contrast)

    def set_padding_ratio(self, ratio):
        self.filter.padding_ratio = ratio

    def set_blend_parameters(self, alpha, enable):
        self.filter.blend_alpha = alpha
        self.filter.enable_blend = enable

    def set_noise_type(self, ntype):
        self.noise_gen.set_noise_type(ntype)

    def set_gaussian_noise_parameters(self, mean, std):
        self.noise_gen.gaussian_mean = mean
        self.noise_gen.gaussian_std = std

    def set_gpu_mode(self, enabled):
        self.use_gpu = enabled

    def auto_optimize_parameters(self, image):
        d0, n = self.optimizer.optimize(image)
        self.filter.set_params(self.filter.filter_type, d0, n)
        return True, d0, n

    @property
    def frequency_filter(self):
        return self.filter

    @property
    def noise_generator(self):
        return self.noise_gen

    @property
    def detail_enhancer(self):
        return self.enhancer
