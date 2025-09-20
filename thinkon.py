"""
Fractal Thinkon 框架 - 门面模块

提供统一的导入接口，兼容现有测试和文档
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "Fractal Think Working Group"

# 导出所有核心组件
from thinkon_core import *

# 确保版本信息被导出
__all__ = getattr(__import__('thinkon_core'), '__all__', []) + ['__version__', '__author__']