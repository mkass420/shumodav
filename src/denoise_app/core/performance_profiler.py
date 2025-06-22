"""
Модуль для профилирования производительности GPU ускорения
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple
import psutil
import threading
from .image_processor import ImageProcessor


class PerformanceProfiler:
    """Профилировщик производительности GPU ускорения"""
    
    def __init__(self):
        self.measurements = []
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict:
        """Получение информации о системе"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'gpu_available': False,
            'gpu_type': 'None'
        }
        
        # Проверяем CUDA
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count > 0:
                info['gpu_available'] = True
                info['gpu_type'] = 'CUDA'
                info['gpu_count'] = device_count
        except:
            pass
        
        # Проверяем OpenCL
        if not info['gpu_available']:
            try:
                if cv2.ocl.useOpenCL():
                    info['gpu_available'] = True
                    info['gpu_type'] = 'OpenCL'
            except:
                pass
        
        return info
    
    def benchmark_processing(self, image: np.ndarray, iterations: int = 5) -> Dict:
        """
        Бенчмарк обработки изображения
        
        Args:
            image: Тестовое изображение
            iterations: Количество итераций для усреднения
            
        Returns:
            Результаты бенчмарка
        """
        print(f"DEBUG: PerformanceProfiler.benchmark_processing: начинаем бенчмарк с {iterations} итерациями")
        
        results = {
            'cpu_times': [],
            'gpu_times': [],
            'cpu_memory': [],
            'gpu_memory': [],
            'speedup': 0.0,
            'memory_usage': 0.0
        }
        
        # CPU бенчмарк
        print("DEBUG: PerformanceProfiler.benchmark_processing: тестируем CPU")
        for i in range(iterations):
            start_time = time.time()
            start_memory = psutil.virtual_memory().used
            
            processor = ImageProcessor()
            processor.set_gpu_mode(False)
            result = processor.process_image(image)
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            cpu_time = end_time - start_time
            cpu_memory = end_memory - start_memory
            
            results['cpu_times'].append(cpu_time)
            results['cpu_memory'].append(cpu_memory)
            
            print(f"DEBUG: PerformanceProfiler.benchmark_processing: CPU итерация {i+1}: {cpu_time:.3f}с, память: {cpu_memory/1024/1024:.1f}MB")
        
        # GPU бенчмарк (если доступен)
        if self.system_info['gpu_available']:
            print(f"DEBUG: PerformanceProfiler.benchmark_processing: тестируем {self.system_info['gpu_type']}")
            for i in range(iterations):
                start_time = time.time()
                start_memory = psutil.virtual_memory().used
                
                processor = ImageProcessor()
                processor.set_gpu_mode(True)
                result = processor.process_image(image)
                
                end_time = time.time()
                end_memory = psutil.virtual_memory().used
                
                gpu_time = end_time - start_time
                gpu_memory = end_memory - start_memory
                
                results['gpu_times'].append(gpu_time)
                results['gpu_memory'].append(gpu_memory)
                
                print(f"DEBUG: PerformanceProfiler.benchmark_processing: GPU итерация {i+1}: {gpu_time:.3f}с, память: {gpu_memory/1024/1024:.1f}MB")
        
        # Вычисляем средние значения
        avg_cpu_time = np.mean(results['cpu_times'])
        avg_cpu_memory = np.mean(results['cpu_memory'])
        
        if results['gpu_times']:
            avg_gpu_time = np.mean(results['gpu_times'])
            avg_gpu_memory = np.mean(results['gpu_memory'])
            
            results['speedup'] = avg_cpu_time / avg_gpu_time
            results['memory_usage'] = avg_gpu_memory / avg_cpu_memory
            results['avg_cpu_time'] = avg_cpu_time
            results['avg_gpu_time'] = avg_gpu_time
            results['avg_cpu_memory'] = avg_cpu_memory
            results['avg_gpu_memory'] = avg_gpu_memory
        else:
            results['avg_cpu_time'] = avg_cpu_time
            results['avg_cpu_memory'] = avg_cpu_memory
        
        print(f"DEBUG: PerformanceProfiler.benchmark_processing: результаты бенчмарка:")
        print(f"  CPU время: {avg_cpu_time:.3f}с")
        if results['gpu_times']:
            print(f"  GPU время: {avg_gpu_time:.3f}с")
            print(f"  Ускорение: {results['speedup']:.2f}x")
            print(f"  Использование памяти: {results['memory_usage']:.2f}x")
        
        return results
    
    def profile_memory_usage(self, image: np.ndarray) -> Dict:
        """
        Профилирование использования памяти
        
        Args:
            image: Тестовое изображение
            
        Returns:
            Результаты профилирования памяти
        """
        print("DEBUG: PerformanceProfiler.profile_memory_usage: профилируем использование памяти")
        
        # Начальное состояние памяти
        initial_memory = psutil.virtual_memory()
        
        # CPU обработка
        processor_cpu = ImageProcessor()
        processor_cpu.set_gpu_mode(False)
        
        memory_before_cpu = psutil.virtual_memory().used
        result_cpu = processor_cpu.process_image(image)
        memory_after_cpu = psutil.virtual_memory().used
        
        cpu_memory_usage = memory_after_cpu - memory_before_cpu
        
        # GPU обработка (если доступен)
        gpu_memory_usage = 0
        if self.system_info['gpu_available']:
            processor_gpu = ImageProcessor()
            processor_gpu.set_gpu_mode(True)
            
            memory_before_gpu = psutil.virtual_memory().used
            result_gpu = processor_gpu.process_image(image)
            memory_after_gpu = psutil.virtual_memory().used
            
            gpu_memory_usage = memory_after_gpu - memory_before_gpu
        
        # Финальное состояние памяти
        final_memory = psutil.virtual_memory()
        
        results = {
            'initial_memory': initial_memory.used,
            'final_memory': final_memory.used,
            'cpu_memory_usage': cpu_memory_usage,
            'gpu_memory_usage': gpu_memory_usage,
            'total_memory_change': final_memory.used - initial_memory.used,
            'memory_efficiency': cpu_memory_usage / gpu_memory_usage if gpu_memory_usage > 0 else float('inf')
        }
        
        print(f"DEBUG: PerformanceProfiler.profile_memory_usage: результаты профилирования памяти:")
        print(f"  CPU память: {cpu_memory_usage/1024/1024:.1f}MB")
        if gpu_memory_usage > 0:
            print(f"  GPU память: {gpu_memory_usage/1024/1024:.1f}MB")
            print(f"  Эффективность памяти: {results['memory_efficiency']:.2f}x")
        
        return results
    
    def get_optimization_recommendations(self, benchmark_results: Dict) -> List[str]:
        """
        Получение рекомендаций по оптимизации
        
        Args:
            benchmark_results: Результаты бенчмарка
            
        Returns:
            Список рекомендаций
        """
        recommendations = []
        
        if benchmark_results['speedup'] < 1.5:
            recommendations.append("GPU ускорение показывает низкую эффективность. Рассмотрите оптимизацию алгоритмов.")
        
        if benchmark_results['memory_usage'] > 2.0:
            recommendations.append("GPU использует много памяти. Рассмотрите уменьшение размера батча или оптимизацию памяти.")
        
        if benchmark_results['speedup'] > 3.0:
            recommendations.append("Отличное GPU ускорение! Рассмотрите увеличение размера батча для еще большей эффективности.")
        
        if not self.system_info['gpu_available']:
            recommendations.append("GPU недоступен. Установите драйверы CUDA или OpenCL для ускорения.")
        
        return recommendations
    
    def generate_report(self, image: np.ndarray) -> str:
        """
        Генерация отчета о производительности
        
        Args:
            image: Тестовое изображение
            
        Returns:
            Текстовый отчет
        """
        print("DEBUG: PerformanceProfiler.generate_report: генерируем отчет о производительности")
        
        # Выполняем бенчмарк
        benchmark_results = self.benchmark_processing(image)
        
        # Профилируем память
        memory_results = self.profile_memory_usage(image)
        
        # Получаем рекомендации
        recommendations = self.get_optimization_recommendations(benchmark_results)
        
        # Формируем отчет
        report = f"""
ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ GPU УСКОРЕНИЯ
========================================

ИНФОРМАЦИЯ О СИСТЕМЕ:
- CPU ядер: {self.system_info['cpu_count']}
- Общая память: {self.system_info['memory_total']/1024/1024/1024:.1f}GB
- GPU доступен: {'Да' if self.system_info['gpu_available'] else 'Нет'}
- Тип GPU: {self.system_info['gpu_type']}

РЕЗУЛЬТАТЫ БЕНЧМАРКА:
- Среднее время CPU: {benchmark_results.get('avg_cpu_time', 0):.3f}с
"""
        
        if benchmark_results.get('avg_gpu_time'):
            report += f"- Среднее время GPU: {benchmark_results['avg_gpu_time']:.3f}с\n"
            report += f"- Ускорение: {benchmark_results['speedup']:.2f}x\n"
            report += f"- Использование памяти: {benchmark_results['memory_usage']:.2f}x\n"
        
        report += f"""
ПРОФИЛИРОВАНИЕ ПАМЯТИ:
- Использование памяти CPU: {memory_results['cpu_memory_usage']/1024/1024:.1f}MB
"""
        
        if memory_results['gpu_memory_usage'] > 0:
            report += f"- Использование памяти GPU: {memory_results['gpu_memory_usage']/1024/1024:.1f}MB\n"
            report += f"- Эффективность памяти: {memory_results['memory_efficiency']:.2f}x\n"
        
        report += f"""
РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:
"""
        
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report 