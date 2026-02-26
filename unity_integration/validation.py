"""
Input Validation Utilities for TwinBrain Framework
==================================================

Provides validation functions to ensure data integrity and security.
"""

from pathlib import Path
from typing import List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_region_ids(
    region_ids: List[int],
    n_regions: int = 200,
    allow_empty: bool = False
) -> List[int]:
    """
    Validate and filter region IDs.
    
    Args:
        region_ids: List of region IDs to validate
        n_regions: Maximum number of regions
        allow_empty: Whether to allow empty list
    
    Returns:
        List of valid region IDs
    
    Raises:
        ValidationError: If validation fails
    
    Example:
        >>> validate_region_ids([1, 5, 10], n_regions=200)
        [1, 5, 10]
        >>> validate_region_ids([1, 999], n_regions=200)
        [1]  # 999 is filtered out
    """
    if not isinstance(region_ids, list):
        raise ValidationError(f"region_ids must be a list, got {type(region_ids)}")
    
    if not region_ids and not allow_empty:
        raise ValidationError("region_ids cannot be empty")
    
    # Filter valid region IDs
    valid_ids = [r for r in region_ids if isinstance(r, int) and 0 <= r < n_regions]
    
    # Log filtered IDs
    invalid_ids = set(region_ids) - set(valid_ids)
    if invalid_ids:
        logger.warning(
            f"Filtered out invalid region IDs: {invalid_ids}. "
            f"Valid range is [0, {n_regions-1}]"
        )
    
    if not valid_ids and not allow_empty:
        raise ValidationError(
            f"No valid region IDs after filtering. "
            f"Valid range is [0, {n_regions-1}]"
        )
    
    return valid_ids


def validate_amplitude(
    amplitude: float,
    min_amp: float = 0.0,
    max_amp: float = 10.0
) -> float:
    """
    Validate stimulation amplitude.
    
    Args:
        amplitude: Amplitude value to validate
        min_amp: Minimum allowed amplitude (0.0 = no stimulation, valid for counterfactual)
        max_amp: Maximum allowed amplitude
    
    Returns:
        Validated amplitude
    
    Raises:
        ValidationError: If amplitude is out of range
    
    Example:
        >>> validate_amplitude(0.5)
        0.5
        >>> validate_amplitude(20.0)
        ValidationError: Amplitude must be between 0.0 and 10.0
    """
    if not isinstance(amplitude, (int, float)):
        raise ValidationError(f"amplitude must be numeric, got {type(amplitude)}")
    
    if not min_amp <= amplitude <= max_amp:
        raise ValidationError(
            f"Amplitude must be between {min_amp} and {max_amp}, got {amplitude}"
        )
    
    return float(amplitude)


def validate_pattern(pattern: str, allowed_patterns: Optional[List[str]] = None) -> str:
    """
    Validate stimulation pattern.
    
    Args:
        pattern: Pattern name to validate
        allowed_patterns: List of allowed patterns (default: sine, pulse, ramp, constant)
    
    Returns:
        Validated pattern name
    
    Raises:
        ValidationError: If pattern is not allowed
    
    Example:
        >>> validate_pattern("sine")
        'sine'
        >>> validate_pattern("invalid")
        ValidationError: Pattern must be one of ...
    """
    if allowed_patterns is None:
        allowed_patterns = ["sine", "pulse", "ramp", "constant"]
    
    if not isinstance(pattern, str):
        raise ValidationError(f"pattern must be a string, got {type(pattern)}")
    
    pattern = pattern.lower().strip()
    
    if pattern not in allowed_patterns:
        raise ValidationError(
            f"Pattern must be one of {allowed_patterns}, got '{pattern}'"
        )
    
    return pattern


def validate_frequency(
    frequency: float,
    min_freq: float = 0.1,
    max_freq: float = 100.0
) -> float:
    """
    Validate stimulation frequency.
    
    Args:
        frequency: Frequency value to validate (Hz)
        min_freq: Minimum allowed frequency
        max_freq: Maximum allowed frequency
    
    Returns:
        Validated frequency
    
    Raises:
        ValidationError: If frequency is out of range
    """
    if not isinstance(frequency, (int, float)):
        raise ValidationError(f"frequency must be numeric, got {type(frequency)}")
    
    if not min_freq <= frequency <= max_freq:
        raise ValidationError(
            f"Frequency must be between {min_freq} and {max_freq} Hz, got {frequency}"
        )
    
    return float(frequency)


def validate_duration(
    duration: int,
    min_duration: int = 1,
    max_duration: int = 1000
) -> int:
    """
    Validate stimulation duration.
    
    Args:
        duration: Duration in time steps
        min_duration: Minimum allowed duration
        max_duration: Maximum allowed duration
    
    Returns:
        Validated duration
    
    Raises:
        ValidationError: If duration is out of range
    """
    if not isinstance(duration, int):
        raise ValidationError(f"duration must be an integer, got {type(duration)}")
    
    if not min_duration <= duration <= max_duration:
        raise ValidationError(
            f"Duration must be between {min_duration} and {max_duration} steps, "
            f"got {duration}"
        )
    
    return duration


def validate_path(
    path: Union[str, Path],
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    base_dir: Optional[Path] = None
) -> Path:
    """
    Validate file/directory path.
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
        must_be_file: Whether path must be a file
        must_be_dir: Whether path must be a directory
        base_dir: Base directory (path must be within this directory for security)
    
    Returns:
        Validated Path object
    
    Raises:
        ValidationError: If validation fails
    
    Example:
        >>> validate_path("/path/to/file.txt", must_exist=True)
        Path('/path/to/file.txt')
    """
    if not isinstance(path, (str, Path)):
        raise ValidationError(f"path must be string or Path, got {type(path)}")
    
    path = Path(path)
    
    # Check if path is within base_dir (prevent path traversal)
    if base_dir is not None:
        base_dir = Path(base_dir).resolve()
        try:
            resolved_path = path.resolve()
            if not str(resolved_path).startswith(str(base_dir)):
                raise ValidationError(
                    f"Path must be within {base_dir}, got {resolved_path}"
                )
        except (OSError, RuntimeError) as e:
            raise ValidationError(f"Invalid path: {e}")
    
    if must_exist and not path.exists():
        raise ValidationError(f"Path does not exist: {path}")
    
    if must_be_file and path.exists() and not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")
    
    if must_be_dir and path.exists() and not path.is_dir():
        raise ValidationError(f"Path is not a directory: {path}")
    
    return path


def validate_n_steps(
    n_steps: int,
    min_steps: int = 1,
    max_steps: int = 1000
) -> int:
    """
    Validate number of prediction/simulation steps.
    
    Args:
        n_steps: Number of steps
        min_steps: Minimum allowed steps
        max_steps: Maximum allowed steps
    
    Returns:
        Validated number of steps
    
    Raises:
        ValidationError: If n_steps is out of range
    """
    if not isinstance(n_steps, int):
        raise ValidationError(f"n_steps must be an integer, got {type(n_steps)}")
    
    if not min_steps <= n_steps <= max_steps:
        raise ValidationError(
            f"n_steps must be between {min_steps} and {max_steps}, got {n_steps}"
        )
    
    return n_steps


def validate_json_data(data: Any, schema: Optional[dict] = None) -> bool:
    """
    Validate JSON data against a schema.
    
    Args:
        data: Data to validate
        schema: JSON schema (if None, just check if serializable)
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If validation fails
    
    Example:
        >>> validate_json_data({"key": "value"})
        True
    """
    import json
    
    # Check if JSON serializable
    try:
        json.dumps(data)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Data is not JSON serializable: {e}")
    
    # TODO: Add schema validation using jsonschema library if needed
    if schema is not None:
        logger.warning("Schema validation not yet implemented")
    
    return True


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to range [min_val, max_val].
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clamped value
    
    Example:
        >>> clamp(15, 0, 10)
        10
        >>> clamp(-5, 0, 10)
        0
    """
    return max(min_val, min(max_val, value))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    
    Example:
        >>> sanitize_filename("../../../etc/passwd")
        'etc_passwd'
    """
    # Remove path components
    filename = Path(filename).name
    
    # Replace unsafe characters
    unsafe_chars = ['/', '\\', '..', ':', '*', '?', '"', '<', '>', '|']
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = name[:max_length - len(ext)] + ext
    
    return filename


# Example usage and tests
if __name__ == "__main__":
    # Test region ID validation
    try:
        valid_regions = validate_region_ids([1, 5, 10, 999], n_regions=200)
        print(f"Valid regions: {valid_regions}")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Test amplitude validation
    try:
        amp = validate_amplitude(0.5)
        print(f"Valid amplitude: {amp}")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Test pattern validation
    try:
        pattern = validate_pattern("sine")
        print(f"Valid pattern: {pattern}")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Test path validation
    try:
        path = validate_path(__file__, must_exist=True, must_be_file=True)
        print(f"Valid path: {path}")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Test filename sanitization
    unsafe_name = "../../../etc/passwd"
    safe_name = sanitize_filename(unsafe_name)
    print(f"Sanitized: '{unsafe_name}' -> '{safe_name}'")
