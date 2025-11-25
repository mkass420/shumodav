import numpy as np
import pywt
import cv2


class WaveletProcessor:
    def __init__(self):
        self.wavelet_name = "db4"
        self.level = 2
        self.threshold = 0.1
        self.mode = "soft"

        # Режимы
        self.use_swt = False
        self.use_post_smoothing = False

        # Сила подавления цветного шума (0.0 - 1.0)
        self.chroma_strength = 0.5

        self._coeffs_cache = None

    def process_image(self, image):
        if image is None:
            return None

        if self.use_swt:
            result = self._process_swt(image)
        else:
            result = self._process_dwt(image)

        if self.use_post_smoothing:
            result = self._apply_bilateral_smoothing(result)

        return result

    def _process_swt(self, image):
        is_color = len(image.shape) == 3
        if is_color:
            ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            channels = [ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]]
        else:
            channels = [image]

        restored_channels = []
        h, w = channels[0].shape
        factor = 2**self.level
        pad_h = (factor - (h % factor)) % factor
        pad_w = (factor - (w % factor)) % factor

        for idx, channel in enumerate(channels):
            if pad_h > 0 or pad_w > 0:
                channel_padded = np.pad(channel, ((0, pad_h), (0, pad_w)), "edge")
            else:
                channel_padded = channel

            coeffs = pywt.swt2(channel_padded, self.wavelet_name, level=self.level)

            is_chroma = idx > 0

            if is_chroma:
                thresh_boost = 1.0 + (self.chroma_strength * 5.0)
                current_threshold = self.threshold * thresh_boost
            else:
                current_threshold = self.threshold

            clean_coeffs = self._apply_threshold_swt(
                coeffs, current_threshold, is_chroma
            )

            rec = pywt.iswt2(clean_coeffs, self.wavelet_name)
            rec = rec[:h, :w]
            restored_channels.append(rec)

        if is_color:
            merged_ycbcr = np.dstack(restored_channels)
            result = cv2.cvtColor(merged_ycbcr, cv2.COLOR_YCrCb2RGB)
        else:
            result = restored_channels[0]

        return np.clip(result, 0.0, 1.0)

    def _process_dwt(self, image):
        is_color = len(image.shape) == 3
        if is_color:
            ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            channels = [ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]]
        else:
            channels = [image]

        restored_channels = []
        for idx, channel in enumerate(channels):
            coeffs = pywt.wavedec2(channel, self.wavelet_name, level=self.level)

            is_chroma = idx > 0

            # Та же логика буста порога для DWT
            if is_chroma:
                thresh_boost = 1.0 + (self.chroma_strength * 5.0)
                current_threshold = self.threshold * thresh_boost
            else:
                current_threshold = self.threshold

            clean_coeffs = self._apply_threshold_dwt(
                coeffs, current_threshold, is_chroma
            )

            rec = pywt.waverec2(clean_coeffs, self.wavelet_name)
            h, w = channel.shape
            rec = rec[:h, :w]
            restored_channels.append(rec)

        if is_color:
            merged_ycbcr = np.dstack(restored_channels)
            result = cv2.cvtColor(merged_ycbcr, cv2.COLOR_YCrCb2RGB)
        else:
            result = restored_channels[0]
        return np.clip(result, 0.0, 1.0)

    def _apply_threshold_dwt(self, coeffs, threshold, is_chroma):
        clean_coeffs = [coeffs[0]]  # LL

        # Дополнительное подавление амплитуды (Desaturation)
        # Если слайдер на максимуме, suppression -> 0 (полное обесцвечивание остатков)
        suppression = (1.0 - self.chroma_strength * 0.8) if is_chroma else 1.0

        for detail_tuple in coeffs[1:]:
            clean_details = tuple(
                pywt.threshold(d, threshold, mode=self.mode) * suppression
                for d in detail_tuple
            )
            clean_coeffs.append(clean_details)
        return clean_coeffs

    def _apply_threshold_swt(self, coeffs, threshold, is_chroma):
        clean_coeffs = []

        suppression = (1.0 - self.chroma_strength * 0.8) if is_chroma else 1.0

        for level_data in coeffs:
            cA, (cH, cV, cD) = level_data

            cH = pywt.threshold(cH, threshold, mode=self.mode) * suppression
            cV = pywt.threshold(cV, threshold, mode=self.mode) * suppression
            cD = pywt.threshold(cD, threshold, mode=self.mode) * suppression

            clean_coeffs.append((cA, (cH, cV, cD)))
        return clean_coeffs

    def _apply_bilateral_smoothing(self, image):
        smoothed = cv2.bilateralFilter(image, d=5, sigmaColor=0.08, sigmaSpace=5)
        return smoothed

    def generate_mosaic(self, image):
        if image is None:
            return None
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        coeffs = pywt.wavedec2(gray, self.wavelet_name, level=self.level)
        return self._coeffs_to_image(coeffs)

    def _coeffs_to_image(self, coeffs):
        def normalize(arr):
            arr = np.abs(arr)
            arr = np.log(arr + 1e-5)
            mn, mx = arr.min(), arr.max()
            if mx - mn == 0:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        mosaic = normalize(coeffs[0])
        for i, (LH, HL, HH) in enumerate(coeffs[1:]):
            LH_img = normalize(LH)
            HL_img = normalize(HL)
            HH_img = normalize(HH)
            target_h, target_w = LH_img.shape
            mosaic = cv2.resize(
                mosaic, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
            top = np.hstack((mosaic, LH_img))
            bottom = np.hstack((HL_img, HH_img))
            mosaic = np.vstack((top, bottom))
        return mosaic

    def set_params(
        self,
        wavelet=None,
        level=None,
        threshold=None,
        mode=None,
        post_smooth=None,
        use_swt=None,
        chroma_str=None,
    ):
        if wavelet:
            self.wavelet_name = wavelet
        if level:
            self.level = int(level)
        if threshold is not None:
            self.threshold = float(threshold)
        if mode:
            self.mode = mode
        if post_smooth is not None:
            self.use_post_smoothing = post_smooth
        if use_swt is not None:
            self.use_swt = use_swt
        if chroma_str is not None:
            self.chroma_strength = float(chroma_str)
