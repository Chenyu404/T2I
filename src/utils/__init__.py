"""
工具模块
提供项目通用的工具函数
"""

from .device_utils import get_device, setup_device
from .server_utils import is_server_environment, setup_server_matplotlib

__all__ = [
    'get_device',
    'setup_device', 
    'is_server_environment',
    'setup_server_matplotlib'
]
