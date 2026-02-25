import psutil
import logging

logger = logging.getLogger(__name__)

def check_memory() -> dict:
    """
    Check memory and provide status and recommendations.
    Uses available_gb with realistic thresholds for edge devices.
    """
    mem = psutil.virtual_memory()
    used_gb = (mem.total - mem.available) / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)

    status = {
        'used_gb': used_gb,
        'available_gb': available_gb,
        'percent': mem.percent,
        'can_load_model': available_gb > 2.0,
        'can_process_request': available_gb > 1.0,
        'recommended_action': 'none',
        'status': 'ok'
    }

    if available_gb < 0.5:  # Hard limit
        status['status'] = 'critical'
        status['can_load_model'] = False
        status['can_process_request'] = False
        status['recommended_action'] = 'reject_request'
        logger.critical(f"CRITICAL: Only {available_gb:.1f}GB available. Rejecting new operations.")
    elif available_gb < 1.0:
        status['status'] = 'warning'
        status['can_load_model'] = False
        status['recommended_action'] = 'avoid_loads'
        logger.warning(f"Low memory: {available_gb:.1f}GB available. Avoiding model loads.")
    else:
        logger.info(f"Memory: {available_gb:.1f}GB available / {mem.total / (1024**3):.1f}GB total")

    return status
