"""
Модуль для батчевой обработки изображений с GPU ускорением
"""

import cv2
import numpy as np
import time
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from .image_processor import ImageProcessor


class BatchProcessor:
    """Батчевая обработка изображений с GPU ускорением"""
    
    def __init__(self, acceleration_type="OpenCL"):
        self.acceleration_type = acceleration_type
        self.use_gpu = acceleration_type != "CPU"
        self.batch_size = 4  # Оптимальный размер батча для GPU
        
    def process_batch(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Батчевая обработка изображений
        
        Args:
            images: Список изображений для обработки
            
        Returns:
            Список кортежей (зашумленное, отфильтрованное, спектры)
        """
        if not images:
            return []
            
        start_time = time.time()
        
        if self.use_gpu:
            results = self._process_batch_gpu(images)
        else:
            results = self._process_batch_cpu(images)
            
        end_time = time.time()
        print(f"DEBUG: BatchProcessor.process_batch: общее время обработки {len(images)} изображений = {end_time - start_time:.3f}с")
        
        return results
    
    def _process_batch_gpu(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """GPU батчевая обработка"""
        if self.acceleration_type == "CUDA":
            return self._process_batch_cuda(images)
        elif self.acceleration_type == "OpenCL":
            return self._process_batch_opencl(images)
        else:
            return self._process_batch_cpu(images)
    
    def _process_batch_cuda(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """CUDA батчевая обработка"""
        try:
            # Разбиваем на батчи оптимального размера
            batches = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]
            results = []
            
            for batch in batches:
                # Обрабатываем батч на GPU
                batch_results = self._process_single_batch_cuda(batch)
                results.extend(batch_results)
                
            return results
            
        except Exception as e:
            print(f"CUDA batch processing failed: {e}, falling back to CPU")
            return self._process_batch_cpu(images)
    
    def _process_single_batch_cuda(self, batch: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Обработка одного батча на CUDA"""
        # Создаем один процессор для всего батча (исправлено для предотвращения утечек)
        processor = ImageProcessor()
        processor.set_gpu_mode(True)
        
        # Обрабатываем последовательно для экономии памяти
        results = []
        for i, image in enumerate(batch):
            try:
                result = processor.process_image(image)
                results.append(result)
                
                # Очищаем память после каждого изображения
                processor.force_memory_cleanup()
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                # Возвращаем пустой результат
                results.append((None, None, (None, None, None)))
        
        # Финальная очистка памяти
        processor.force_memory_cleanup()
        
        return results
    
    def _process_batch_opencl(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """OpenCL батчевая обработка"""
        try:
            # Разбиваем на батчи оптимального размера
            batches = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]
            results = []
            
            for batch in batches:
                # Обрабатываем батч на OpenCL
                batch_results = self._process_single_batch_opencl(batch)
                results.extend(batch_results)
                
            return results
            
        except Exception as e:
            print(f"OpenCL batch processing failed: {e}, falling back to CPU")
            return self._process_batch_cpu(images)
    
    def _process_single_batch_opencl(self, batch: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Обработка одного батча на OpenCL"""
        # Создаем один процессор для всего батча (исправлено для предотвращения утечек)
        processor = ImageProcessor()
        processor.set_gpu_mode(True)
        
        # Обрабатываем последовательно для экономии памяти
        results = []
        for i, image in enumerate(batch):
            try:
                result = processor.process_image(image)
                results.append(result)
                
                # Очищаем память после каждого изображения
                processor.force_memory_cleanup()
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                # Возвращаем пустой результат
                results.append((None, None, (None, None, None)))
        
        # Финальная очистка памяти
        processor.force_memory_cleanup()
        
        return results
    
    def _process_batch_cpu(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """CPU батчевая обработка с многопоточностью"""
        # Создаем процессоры для каждого изображения
        processors = [ImageProcessor() for _ in images]
        
        # Настраиваем CPU режим
        for processor in processors:
            processor.set_gpu_mode(False)
        
        # Обрабатываем с многопоточностью
        with Pool(processes=min(len(images), cpu_count())) as pool:
            # Создаем задачи для обработки
            tasks = [(image, processor) for image, processor in zip(images, processors)]
            results = pool.map(self._process_single_image, tasks)
        
        return results
    
    def _process_single_image(self, task: Tuple[np.ndarray, ImageProcessor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Обработка одного изображения"""
        image, processor = task
        try:
            return processor.process_image(image)
        except Exception as e:
            print(f"Error processing image: {e}")
            return (None, None, (None, None, None))
    
    def set_batch_size(self, batch_size: int):
        """Установка размера батча"""
        self.batch_size = max(1, min(batch_size, 16))  # Ограничиваем размер батча
        print(f"DEBUG: BatchProcessor.set_batch_size: установлен размер батча = {self.batch_size}")
    
    def get_optimal_batch_size(self, image_size: Tuple[int, int]) -> int:
        """Определение оптимального размера батча для данного размера изображения"""
        height, width = image_size
        pixels = height * width
        
        # Эмпирическая формула для определения оптимального размера батча
        if pixels < 1000000:  # < 1MP
            return 8
        elif pixels < 4000000:  # < 4MP
            return 4
        elif pixels < 16000000:  # < 16MP
            return 2
        else:  # >= 16MP
            return 1 