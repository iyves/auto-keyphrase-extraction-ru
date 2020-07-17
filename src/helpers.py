"""Helper functions to be used in multiple jupyter notebooks
"""

from django.utils.text import slugify
from pathlib import Path
from typing import Dict, List, Tuple
import os


def to_file_safe_string(value: str, max_strlen: int = 200) -> str:
    """Parses a string into a valid filename.
    
    Args:
        value:
            The string to parse.
        max_strlen:
            Optional; The maximum length of a file name (not including the file
            extension). Default is 200 characters
        
    Returns:
        A string that conforms to the django standards of a valid URL slug. For example:
        
        "управление-лексиконом-в-онтологической-семантике-p-636.txt"
        
    """
    slug = slugify(value, allow_unicode=True)
    return slug[0:max_strlen]


def get_text(filename: Path) -> str:
    """Get the text from a specified file.
    
    Args:
        filename:
            The file from which to get text.
    
    Returns:
        The text from a specified file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()
  

def get_file_names(root_dir: Path, extension: str) -> List[Path]:
    """Get the path of all files of a specified extension from a specified directory
    
    Args:
        root_dir:
            The directory from which to get file paths.
        extension:
            The extension corresponding to the file type to get.
    """
    file_names = []
    for root_dir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                file_names.append(Path(root_dir, file))
    return file_names