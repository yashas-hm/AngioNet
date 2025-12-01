"""
One-Click Reproducibility Script

Downloads pre-computed outputs, sets up environment, runs pipeline, and launches EDA notebook.
This script enables complete reproduction with a single command.

Usage:
    python oneclick.py
"""
import os
import subprocess
import sys
import urllib.request
import zipfile

# Configuration
DOWNLOAD_URL = "https://github.com/yashas-hm-unc/blood-vessel-segviz/releases/download/v1.0.1/precomputed_outputs.zip"
ARCHIVE_NAME = "precomputed_outputs.zip"
EDA_NOTEBOOK = "eda_angio_net.ipynb"


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_file(url, destination):
    """Download a file from URL with progress indicator."""
    print(f"Downloading from {url}...")

    try:
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 // total_size)
                print(f"\rProgress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("\nDownload complete.")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False


def extract_archive(archive_path, destination):
    """Extract zip archive to destination."""
    print(f"Extracting {archive_path}...")

    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False


def delete_file(file_path):
    """Delete a file."""
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
        return True
    except Exception as e:
        print(f"Error deleting file: {e}")
        return False


def get_venv_python():
    """Get the path to the virtual environment Python."""
    root = get_project_root()
    if sys.platform == 'win32':
        return os.path.join(root, 'venv', 'Scripts', 'python.exe')
    else:
        return os.path.join(root, 'venv', 'bin', 'python')


def run_pipeline():
    """Run the main pipeline script (creates venv and installs dependencies)."""
    root = get_project_root()
    pipeline_script = os.path.join(root, "Runners/run_pipeline.py")

    print("\n" + "=" * 40)
    print("Running Pipeline")
    print("=" * 40)

    # run_pipeline.py will create venv and re-launch itself in it
    result = subprocess.run([sys.executable, pipeline_script], cwd=root)
    return result.returncode == 0


def install_jupyter_kernel():
    """Install the virtual environment as a Jupyter kernel."""
    venv_python = get_venv_python()

    if not os.path.exists(venv_python):
        print("Virtual environment not found. Run pipeline first.")
        return False

    print("\nInstalling Jupyter kernel...")

    # Register the kernel
    result = subprocess.run([
        venv_python, '-m', 'ipykernel', 'install',
        '--user',
        '--name', 'vessel-segmentation',
        '--display-name', 'Vessel Segmentation (venv)'
    ])

    if result.returncode == 0:
        print("Jupyter kernel installed: 'vessel-segmentation'")
        return True
    else:
        print("Warning: Failed to install Jupyter kernel.")
        return False


def run_notebook():
    """Run the EDA notebook and execute all cells."""
    root = get_project_root()
    notebook_path = os.path.join(root, EDA_NOTEBOOK)
    venv_python = get_venv_python()

    if not os.path.exists(notebook_path):
        print(f"Notebook not found: {notebook_path}")
        return False

    print("\n" + "=" * 40)
    print("Running EDA Notebook")
    print("=" * 40)

    # Execute the notebook in place
    print(f"Executing {EDA_NOTEBOOK}...")
    result = subprocess.run([
        venv_python, '-m', 'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--inplace',
        '--ExecutePreprocessor.kernel_name=vessel-segmentation',
        '--ExecutePreprocessor.timeout=600',
        notebook_path
    ], cwd=root)

    if result.returncode == 0:
        print("Notebook executed successfully.")
        return True
    else:
        print("Warning: Notebook execution completed with errors.")
        return False


def launch_jupyter():
    """Launch Jupyter notebook server."""
    root = get_project_root()
    venv_python = get_venv_python()
    notebook_path = os.path.join(root, EDA_NOTEBOOK)

    print("\n" + "=" * 40)
    print("Launching Jupyter Notebook")
    print("=" * 40)

    print(f"Opening {EDA_NOTEBOOK}...")
    print("Press Ctrl+C to stop the notebook server.\n")

    # Launch Jupyter notebook
    subprocess.run([
        venv_python, '-m', 'jupyter', 'notebook',
        notebook_path,
        '--NotebookApp.kernel_name=vessel-segmentation'
    ], cwd=root)


def main():
    root = get_project_root()
    archive_path = os.path.join(root, ARCHIVE_NAME)

    print("=" * 50)
    print("One-Click Reproducibility Script")
    print("=" * 50)
    print("\nThis script will:")
    print("  1. Download pre-computed outputs")
    print("  2. Extract and place files in correct locations")
    print("  3. Run the pipeline (skips existing outputs)")
    print("  4. Execute the EDA notebook")
    print("  5. Launch Jupyter for interactive exploration")
    print()

    # Step 1: Download pre-computed outputs
    print("\n" + "-" * 40)
    print("Step 1: Download Pre-computed Outputs")
    print("-" * 40)

    # Check if outputs already exist
    model_exists = os.path.exists(os.path.join(root, 'Model', 'unet_checkpoint.pth.tar'))
    predictions_exist = os.path.isdir(os.path.join(root, 'Data', 'outputs', 'predictions'))
    features_exist = os.path.isdir(os.path.join(root, 'Data', 'outputs', 'features'))

    if model_exists and predictions_exist and features_exist:
        print("Pre-computed outputs already exist. Skipping download...")
    else:
        # Download
        if not download_file(DOWNLOAD_URL, archive_path):
            print("Failed to download. Continuing with pipeline (will train from scratch)...")
        else:
            # Step 2: Extract archive
            print("\n" + "-" * 40)
            print("Step 2: Extract Archive")
            print("-" * 40)

            if extract_archive(archive_path, root):
                # Step 3: Delete archive
                print("\n" + "-" * 40)
                print("Step 3: Cleanup")
                print("-" * 40)
                delete_file(archive_path)

    # Step 4: Run pipeline
    print("\n" + "-" * 40)
    print("Step 4: Run Pipeline")
    print("-" * 40)

    if not run_pipeline():
        print("Pipeline failed. Exiting...")
        sys.exit(1)

    # Step 5: Install Jupyter kernel
    print("\n" + "-" * 40)
    print("Step 5: Setup Jupyter Kernel")
    print("-" * 40)

    install_jupyter_kernel()

    # Step 6: Run notebook
    print("\n" + "-" * 40)
    print("Step 6: Execute EDA Notebook")
    print("-" * 40)

    run_notebook()

    # Step 7: Launch Jupyter
    print("\n" + "-" * 40)
    print("Step 7: Launch Jupyter")
    print("-" * 40)

    print("\n" + "=" * 50)
    print("One-Click Setup Complete!")
    print("=" * 50)
    print("\nAll outputs generated:")
    print(f"  - Model: Model/unet_checkpoint.pth.tar")
    print(f"  - Predictions: Data/outputs/predictions/")
    print(f"  - Features: Data/outputs/features/")
    print(f"  - EDA Notebook: {EDA_NOTEBOOK} (executed)")
    print()

    # Ask user if they want to launch Jupyter
    try:
        response = input("Launch Jupyter notebook for interactive exploration? [Y/n]: ").strip().lower()
        if response != 'n':
            launch_jupyter()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == '__main__':
    main()