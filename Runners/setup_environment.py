"""
Environment Setup Script

Creates virtual environment and installs dependencies.
Platform-independent (Windows, macOS, Linux).
"""
import os
import subprocess
import sys

# Minimum Python version required
MIN_PYTHON_VERSION = (3, 10)


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_python_version():
    """Check if Python version meets minimum requirements."""
    current_version = sys.version_info[:2]
    if current_version < MIN_PYTHON_VERSION:
        print(f'Error: Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ is required.')
        print(f'Current version: Python {current_version[0]}.{current_version[1]}')
        sys.exit(1)
    print(f'Python version: {current_version[0]}.{current_version[1]} ✓')
    return True


def get_venv_paths(root):
    """Get platform-specific virtual environment paths."""
    venv_path = os.path.join(root, 'venv')

    if sys.platform == 'win32':
        python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
        pip_path = os.path.join(venv_path, 'Scripts', 'pip.exe')
    else:
        python_path = os.path.join(venv_path, 'bin', 'python')
        pip_path = os.path.join(venv_path, 'bin', 'pip')

    return venv_path, python_path, pip_path


def create_virtual_environment(venv_path):
    """Create virtual environment if it doesn't exist."""
    if os.path.isdir(venv_path):
        print(f'Virtual environment already exists at {venv_path}')
        return True

    print('Creating virtual environment...')
    result = subprocess.run([sys.executable, '-m', 'venv', venv_path])

    if result.returncode != 0:
        print('Error: Failed to create virtual environment.')
        return False

    print('Virtual environment created successfully.')
    return True


def install_requirements(pip_path, requirements_path):
    """Install requirements from requirements.txt."""
    if not os.path.isfile(requirements_path):
        print(f'Warning: requirements.txt not found at {requirements_path}')
        return True

    print('Installing dependencies from requirements.txt...')
    result = subprocess.run([pip_path, 'install', '-r', requirements_path])

    if result.returncode != 0:
        print('Warning: Some dependencies may have failed to install.')
        return False

    print('All dependencies installed successfully.')
    return True


def is_running_in_venv(python_path):
    """Check if currently running in the target virtual environment."""
    current_python = os.path.abspath(sys.executable)
    target_python = os.path.abspath(python_path)
    return current_python == target_python


def setup_environment():
    """
    Main setup function.

    Returns:
        tuple: (venv_python_path, is_in_venv) - Path to venv Python and whether we're in venv
    """
    root = get_project_root()
    requirements_path = os.path.join(root, 'requirements.txt')

    # Check Python version
    check_python_version()

    # Get paths
    venv_path, python_path, pip_path = get_venv_paths(root)

    # Create virtual environment
    if not create_virtual_environment(venv_path):
        sys.exit(1)

    # Check if we're in the venv
    in_venv = is_running_in_venv(python_path)

    if in_venv:
        # Install requirements if we're in the venv
        install_requirements(pip_path, requirements_path)

    return python_path, in_venv


def relaunch_in_venv(python_path):
    """Re-launch the calling script in the virtual environment."""
    print('Re-launching in virtual environment...')
    result = subprocess.run([python_path] + sys.argv)
    sys.exit(result.returncode)


if __name__ == '__main__':
    print('=' * 40)
    print('Environment Setup')
    print('=' * 40)

    python_path, in_venv = setup_environment()

    if in_venv:
        print('\n✓ Environment is ready!')
        print(f'  Virtual environment: {os.path.dirname(os.path.dirname(python_path))}')
    else:
        print('\n✓ Virtual environment created.')
        print(f'  Activate with:')
        if sys.platform == 'win32':
            print(f'    venv\\Scripts\\activate')
        else:
            print(f'    source venv/bin/activate')