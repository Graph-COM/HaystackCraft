def parse_token_count(value):
    """Parse token count from string format like '64K', '128K' etc."""
    if isinstance(value, int):
        return value
    
    value = str(value).upper().strip()
    
    # Handle direct integer input
    if value.isdigit():
        return int(value)
    
    # Parse with suffixes
    if value.endswith('K'):
        return int(float(value[:-1]) * 1000)
    else:
        raise ValueError(f"Invalid token count format: {value}. Use formats like '64K', '128K', '1M', or plain integers.")
