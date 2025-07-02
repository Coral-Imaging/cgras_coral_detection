#!/usr/bin/env python3
"""
Prediction pipeline that combines SAHI prediction, COCO fixing, and COCO combining functionality.
Run with: python predict_pipeline.py --config path/to/config.yaml
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
#sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import our class-based modules
from utils.sahi_predict_edits import SahiPredictor
from utils.fix_coco_json_edits import CocoJsonFixer
from utils.combine_coco_edits import CocoCombiner


def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        sys.exit(1)


def run_sahi_prediction(config, logger):
    """Run the SAHI prediction step."""
    logger.info("Starting SAHI prediction step...")
    
    sahi_config = config.get('sahi_predict', {})
    
    # Get SAHI parameters
    data_dir = sahi_config.get('data_dir')
    output_dir = sahi_config.get('output_dir')
    model_path = sahi_config.get('model_path')
    name = sahi_config.get('name', 'default')
    slice_width = sahi_config.get('slice_width', 640)
    slice_height = sahi_config.get('slice_height', 640)
    overlap = sahi_config.get('overlap', 0.5)
    conf_thresh = sahi_config.get('conf_thresh', 0.4)
    device = sahi_config.get('device', 'cuda:0')
    verbose = sahi_config.get('verbose', True)
    
    if not all([data_dir, output_dir, model_path]):
        logger.error("data_dir, output_dir, and model_path must be specified in sahi_predict config")
        sys.exit(1)
    
    # Initialize predictor and run
    predictor = SahiPredictor(verbose=verbose)
    output_path = predictor.run_prediction_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        model_path=model_path,
        name=name,
        slice_width=slice_width,
        slice_height=slice_height,
        overlap=overlap,
        conf_thresh=conf_thresh,
        device=device
    )
    
    logger.info(f"SAHI prediction completed. Output at: {output_path}")
    return output_path


def run_fix_coco(config, input_path, logger):
    """Run the COCO fixing step."""
    logger.info("Starting COCO fixing step...")
    
    fix_config = config.get('fix_coco', {})
    
    # Get fix parameters
    output_file = fix_config.get('output_file', None)
    verbose = fix_config.get('verbose', True)
    
    # Initialize fixer and run
    fixer = CocoJsonFixer(verbose=verbose)
    fixed_path = fixer.fix_coco_json(input_path, output_file)
    
    logger.info(f"COCO fixing completed. Output at: {fixed_path}")
    return fixed_path


def run_combine_coco(config, base_path, additional_path, logger):
    """Run the COCO combining step."""
    logger.info("Starting COCO combining step...")
    
    combine_config = config.get('combine_coco', {})
    
    # Get combine parameters
    output_path = combine_config.get('output_path')
    verbose = combine_config.get('verbose', True)
    
    if not output_path:
        logger.error("output_path must be specified in combine_coco config")
        sys.exit(1)
    
    # Initialize combiner and run
    combiner = CocoCombiner(verbose=verbose)
    combined_path = combiner.combine_datasets_from_files(base_path, additional_path, output_path)
    
    logger.info(f"COCO combining completed. Output at: {combined_path}")
    return combined_path


def run_pipeline(config_path):
    """
    Run the complete prediction pipeline.
    
    Args:
        config_path (str): Path to YAML configuration file
    """
    logger = setup_logging()
    config = load_config(config_path)
    
    pipeline_config = config.get('pipeline', {})
    steps = pipeline_config.get('steps', ['sahi_predict'])
    
    logger.info(f"Starting prediction pipeline with steps: {steps}")
    
    # Track outputs for chaining steps
    sahi_output = None
    fixed_output = None
    combined_output = None
    
    # Step 1: SAHI Prediction (always first if included)
    if 'sahi_predict' in steps:
        sahi_output = run_sahi_prediction(config, logger)
    
    # Step 2: Fix COCO (optional, uses SAHI output or specified input)
    if 'fix_coco' in steps:
        fix_config = config.get('fix_coco', {})
        input_file = fix_config.get('input_file') or sahi_output
        if not input_file:
            logger.error("fix_coco step requires either input_file in config or sahi_predict step to run first")
            sys.exit(1)
        
        fixed_output = run_fix_coco(config, input_file, logger)
    
    # Step 3: Combine COCO (optional, uses fixed output or specified inputs)
    if 'combine_coco' in steps:
        combine_config = config.get('combine_coco', {})
        base_dataset = combine_config.get('base_dataset')
        additional_dataset = combine_config.get('additional_dataset') or fixed_output or sahi_output

        if not all([base_dataset, additional_dataset]):
            logger.error("combine_coco step requires base_dataset and additional_dataset")
            sys.exit(1)
        
        combined_output = run_combine_coco(config, base_dataset, additional_dataset, logger)
    
    # Summary
    logger.info("Pipeline completed successfully!")
    if sahi_output:
        logger.info(f"SAHI predictions: {sahi_output}")
    if fixed_output:
        logger.info(f"Fixed COCO: {fixed_output}")
    if combined_output:
        logger.info(f"Combined COCO: {combined_output}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run prediction pipeline')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file does not exist: {args.config}")
        sys.exit(1)
    
    run_pipeline(args.config)


if __name__ == "__main__":
    main()