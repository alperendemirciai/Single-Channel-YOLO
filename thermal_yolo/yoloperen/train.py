import argparse
from ultralytics import YOLO
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO with custom model, data, and full hyp config")
    parser.add_argument('--model_yaml', type=str, required=True, help='Path to custom model .yaml file')
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to dataset .yaml file')
    parser.add_argument('--hyp_yaml', type=str, required=True, help='Path to hyperparameters/training config .yaml file')
    parser.add_argument('--project_name', type=str, required=True, help='Project folder to save runs')
    parser.add_argument('--version', type=str, required=True, help='Run name inside project folder')

    return parser.parse_args()

def main():
    args = parse_args()

    # Load hyp config
    with open(args.hyp_yaml, 'r') as f:
        config = yaml.safe_load(f)

    # Inject CLI arguments (override those in hyp_yaml)
    config['model'] = args.model_yaml
    config['data'] = args.data_yaml
    config['project'] = args.project_name
    config['name'] = args.version

    # Initialize and train
    model = YOLO(args.model_yaml)
    model.train(**config)

if __name__ == '__main__':
    main()
