# NOTE: ALWAYS RUN THIS FILE FROM THE ROOT


"""
Vessel Segmentation Pipeline Runner

Platform-independent script to run the complete pipeline with checkpoint detection.
Skips stages if outputs already exist (idempotent execution).
Automatically sets up virtual environment and installs dependencies.
"""
import os
import sys

try:
    # When imported from project root
    from Runners.setup_environment import setup_environment, relaunch_in_venv
except ImportError:
    # When running directly from Runners directory
    from setup_environment import setup_environment, relaunch_in_venv


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dir_exists_and_not_empty(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0


# noinspection PyBroadException
def main():
    root = get_project_root()

    print('=' * 40)
    print('Vessel Segmentation Pipeline')
    print('=' * 40)

    # Step 0: Environment Setup
    print('\nStep 0: Environment Setup')
    print('-' * 40)
    python_path, in_venv = setup_environment()

    # Re-launch in venv if not already running in it
    if not in_venv:
        relaunch_in_venv(python_path)

    # Add project root to path so Scripts can be imported
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import scripts after environment is set up
    from Scripts.data_preprocessing import process_data
    from Scripts.feature_extraction.feature_extraction import run_feature_extraction
    from Scripts.predict import run_prediction
    from Scripts.train import train_model

    # Define paths
    preprocessed_dir = os.path.join(root, 'Data/preprocessed')
    model_path = os.path.join(root, 'Model/unet_checkpoint.pth.tar')
    predictions_dir = os.path.join(root, 'Data/outputs/predictions')
    features_dir = os.path.join(root, 'Data/outputs/features')

    # Step 1: Data Preprocessing
    print('\nStep 1: Data Preprocessing')
    print('-' * 40)
    train_exists = dir_exists_and_not_empty(os.path.join(preprocessed_dir, 'train/image'))
    val_exists = dir_exists_and_not_empty(os.path.join(preprocessed_dir, 'validation/image'))
    test_exists = dir_exists_and_not_empty(os.path.join(preprocessed_dir, 'test/image'))

    if train_exists and val_exists and test_exists:
        print('Preprocessed data already exists. Skipping...')
    else:
        print('Running data preprocessing...')
        try:
            process_data(get_project_root())
        except:
            print('Error: Data preprocessing failed.')
            sys.exit(1)

    # Step 2: Training
    print('\nStep 2: Model Training')
    print('-' * 40)
    if os.path.isfile(model_path):
        print(f'Trained model already exists at {model_path}. Skipping...')
    else:
        print('Training model...')
        try:
            train_model(get_project_root())
        except:
            print('Error: Training failed.')
            sys.exit(1)

    # Step 3: Prediction
    print('\nStep 3: Prediction')
    print('-' * 40)
    if dir_exists_and_not_empty(predictions_dir):
        print('Predictions already exist. Skipping...')
    else:
        print('Running predictions...')
        try:
            run_prediction(get_project_root())
        except:
            print('Error: Prediction failed.')
            sys.exit(1)

    # Step 4: Feature Extraction
    print('\nStep 4: Feature Extraction')
    print('-' * 40)
    if dir_exists_and_not_empty(features_dir):
        print('Features already extracted. Skipping...')
    else:
        print('Extracting features...')
        try:
            run_feature_extraction(get_project_root())
        except:
            print('Error: Feature extraction failed.')
            sys.exit(1)

    print('\n' + '=' * 40)
    print('Pipeline completed successfully!')
    print('=' * 40)
    print('\nOutputs:')
    print(f'  - Preprocessed data: {preprocessed_dir}/')
    print(f'  - Trained model: {model_path}')
    print(f'  - Predictions: {predictions_dir}/')
    print(f'  - Features: {features_dir}/')


if __name__ == '__main__':
    main()