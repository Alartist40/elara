import psutil
import logging

logger = logging.getLogger(__name__)

def check_memory() -> dict:
    """
    Check memory and provide status and recommendations.
    4GB target environment.
    """
    mem = psutil.virtual_memory()
    # For many systems, 'available' is more accurate than 'total - used' for what can be allocated
    used_gb = (mem.total - mem.available) / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    percent = mem.percent

    status = {
        'used_gb': used_gb,
        'available_gb': available_gb,
        'percent': percent,
        'can_load_model': True,
        'can_process_request': True,
        'recommended_action': 'none',
        'status': 'ok'
    }

    if used_gb > 3.8:  # Hard limit approaching
        status['status'] = 'critical'
        status['can_load_model'] = False
        status['can_process_request'] = False
        status['recommended_action'] = 'reject_request'
        logger.critical(f"CRITICAL: Memory at {used_gb:.1f}GB. Rejecting new operations.")
    elif used_gb > 3.2:  # Warning zone
        status['status'] = 'warning'
        status['can_load_model'] = False  # Don't load new models
        status['recommended_action'] = 'avoid_loads'
        logger.warning(f"High memory: {used_gb:.1f}GB. Avoiding model loads.")
    else:
        status['status'] = 'ok'
        logger.info(f"Memory usage: {used_gb:.1f}GB / {mem.total / (1024**3):.1f}GB")

    return status
