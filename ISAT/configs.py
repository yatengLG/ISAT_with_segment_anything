import yaml
from enum import Enum
import os

__all__ = ['load_config', 'save_config', 'STATUSMode', 'DRAWMode', 'MAPMode', 'CONTOURMode']

ISAT_ROOT = os.path.split(os.path.abspath(__file__))[0]
"""Project root - ISAT/"""
SOFTWARE_CONFIG_FILE = os.path.join(ISAT_ROOT, 'software.yaml')
"""Software config file - ISAT/software.yaml"""
CONFIG_FILE = os.path.join(ISAT_ROOT, 'isat.yaml')
"""Category config file - ISAT/isat.yaml"""
CHECKPOINT_PATH = os.path.join(ISAT_ROOT, 'checkpoints')
"""Checkpoints save root - ISAT/checkpoints"""

os.makedirs(os.path.join(CHECKPOINT_PATH, 'tmp'), exist_ok=True)

if not os.path.exists(SOFTWARE_CONFIG_FILE):
    with open(SOFTWARE_CONFIG_FILE, 'w') as f:
        pass

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'w') as f:
        pass

def load_config(file: str) -> dict:
    r"""
    Load config file

    Arguments:
        file (str): config file path

    Returns:
        dict: config dict
    """
    with open(file, 'rb')as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg

def save_config(cfg: dict, file: str) -> None:
    """
    Save config file

    Arguments:
        cfg (dict): config dict
        file (str): config file path
    """
    s = yaml.dump(cfg, allow_unicode=True)
    with open(file, 'w', encoding='utf-8') as f:
        f.write(s)

class STATUSMode(Enum):
    """The status mode"""
    VIEW = 0
    CREATE = 1
    EDIT = 2
    REPAINT = 3

class DRAWMode(Enum):
    """The draw mode."""
    POLYGON = 0
    """Manually draw polygon"""
    SEGMENTANYTHING = 1
    """Segment anything with point prompt."""
    SEGMENTANYTHING_BOX = 2
    """Segment anything with box prompt."""

class MAPMode(Enum):
    """Canvas show map mode"""
    LABEL = 0
    SEMANTIC = 1
    INSTANCE = 2

class CONTOURMode(Enum):
    """
    Contour Mode - ways to convert masks to polygons.
    """
    SAVE_MAX_ONLY = 0
    """Only save max contour."""
    SAVE_EXTERNAL = 1
    """Only save external contour."""
    SAVE_ALL = 2
    """Only save all contour."""

