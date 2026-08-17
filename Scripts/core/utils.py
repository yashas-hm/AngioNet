def print_progress(current, total, prefix='', suffix='', length=30):
    """Print a simple progress bar."""
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    print(f'\r{prefix} |{bar}| {current}/{total} {suffix}', end='', flush=True)
    if current == total:
        print()
