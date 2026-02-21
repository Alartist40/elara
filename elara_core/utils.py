import psutil
import logging

logger = logging.getLogger(__name__)

def check_memory():
    """Warn if approaching 4GB limit."""
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)

    if used_gb > 3.5:
        logger.warning(f"High memory usage: {used_gb:.1f}GB / {total_gb:.1f}GB. Approaching 4GB limit.")
    else:
        logger.info(f"Memory usage: {used_gb:.1f}GB / {total_gb:.1f}GB")

    return used_gb
