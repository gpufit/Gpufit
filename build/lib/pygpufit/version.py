"""
Current version (short and full names).
"""

__version_short__ = '1.2'
__version_full__ = '1.2.0'

__version__ = __version_full__

v_split = __version__.split('.')
__version_tuple__ = tuple(map(int, v_split))