import platform
from constants import DEVICE
import torch

# Check if we're on Mac ARM where OpenVINO is not supported
_is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"

ov = None
core = None

if not _is_mac_arm:
    try:
        import openvino as ov
        core = ov.Core()
    except Exception as e:
        print(f"OpenVINO not available: {e}")
        ov = None
        core = None


def is_openvino_available() -> bool:
    """Check if OpenVINO is available on this system."""
    return ov is not None and core is not None


def is_openvino_device() -> bool:
    """Check if the current DEVICE setting requires OpenVINO."""
    if not is_openvino_available():
        return False
    if DEVICE.lower() == "cpu" or DEVICE.lower()[0] == "g" or DEVICE.lower()[0] == "n":
        return True
    else:
        return False


def get_device_name() -> str:
    if DEVICE == "cuda":
        default_gpu_index = torch.cuda.current_device()
        return torch.cuda.get_device_name(default_gpu_index)
    elif DEVICE == "mps":
        return f"Apple {platform.processor()} (MPS)"
    elif platform.system().lower() == "darwin":
        return platform.processor()
    elif is_openvino_device() and core is not None:
        return core.get_property(DEVICE.upper(), "FULL_DEVICE_NAME")
    else:
        return "CPU"
