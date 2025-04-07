#!/usr/bin/env python3

"""
experiment_runner.py
Runs the complete coral detection scaling experiment with variable numbers of training images.
Coordinates image selection, data processing, model training, and evaluation.
"""

import os
import sys
import yaml
import argparse
import subprocess
import time
import logging
from pathlib import Path
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our image selector
from utils.image_sampler import ImageSelector

def setup_logging(log_level=logging.INFO, experiment_name="CoralScaling"):
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    return logging.getLogger("experiment_runner")

def run_command(cmd, logger, cwd=None):
    """Run a shell command and log output."""
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        
        # Stream output to logger
        for line in process.stdout:
            logger.info(line.strip())
            
        for line in process.stderr:
            logger.error(line.strip())
            
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            return False
            
        return True
        
    except Exception as e:
        logger.exception(f"Error running command: {e}")
        return False

def process_dataset(pipeline_config_path, logger):
    """Process a dataset through the image processing pipeline."""
    logger.info(f"Processing dataset with config: {pipeline_config_path}")
    
    # Run the image processing pipeline
    cmd = [
        "python3",
        "image_processing.py",
        "--config", str(pipeline_config_path)
    ]
    
    return run_command(cmd, logger)

def train_model(train_config_path, logger):
    """Train a model with the specified configuration."""
    logger.info(f"Training model with config: {train_config_path}")
    
    # Run the training script
    cmd = [
        "python3",
        "train.py",
        "--config", str(train_config_path)
    ]
    
    return run_command(cmd, logger)

def collect_metrics(experiment_path, logger):
    """Collect metrics from training runs."""
    logger.info("Collecting metrics from all training runs")
    
    metrics = {}
    
    # Find all training output directories
    for train_dir in experiment_path.glob("outputs/*filtered_split_tiled_balanced"):
        # Extract count from directory name
        try:
            dir_name = train_dir.name
            count = int(dir_name.split('_')[1])  # Assumes format: CoralScaling_10_filtered_split_tiled_balanced
        except (ValueError, IndexError):
            logger.warning(f"Could not extract count from directory name: {dir_name}")
            continue
            
        # Find results.csv from training
        results_path = list(Path(train_dir).glob("**/results.csv"))
        if not results_path:
            logger.warning(f"No results.csv found for count {count}")
            continue
            
        # Get the most recent results file
        results_path = sorted(results_path)[-1]
        
        # Parse metrics
        try:
            with open(results_path, 'r') as f:
                # Skip header
                next(f)
                # Get the last line (final metrics)
                for line in f:
                    last_line = line
                    
                # Parse metrics from last line
                cols = last_line.strip().split(',')
                metrics[count] = {
                    'precision': float(cols[4]),
                    'recall': float(cols[5]),
                    'mAP50': float(cols[6]),
                    'mAP50-95': float(cols[7])
                }
                logger.info(f"Collected metrics for count {count}: {metrics[count]}")
                
        except Exception as e:
            logger.exception(f"Error parsing metrics for count {count}: {e}")
            
    # Save metrics to JSON
    metrics_path = experiment_path / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    logger.info(f"Saved metrics to {metrics_path}")
    return metrics

def generate_plots(metrics, experiment_path, logger):
    """Generate plots from collected metrics."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        logger.info("Generating performance plots")
        
        # Convert metrics to lists for plotting
        counts = sorted(metrics.keys())
        precision = [metrics[c]['precision'] for c in counts]
        recall = [metrics[c]['recall'] for c in counts]
        map50 = [metrics[c]['mAP50'] for c in counts]
        map50_95 = [metrics[c]['mAP50-95'] for c in counts]
        
        # Create plots directory
        plots_dir = experiment_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Overall performance plot
        plt.figure(figsize=(10, 6))
        plt.plot(counts, precision, 'o-', label='Precision')
        plt.plot(counts, recall, 'o-', label='Recall')
        plt.plot(counts, map50, 'o-', label='mAP50')
        plt.plot(counts, map50_95, 'o-', label='mAP50-95')
        
        plt.xscale('log')
        plt.xlabel('Number of Training Images')
        plt.ylabel('Metric Value')
        plt.title('Coral Detection Performance vs. Training Set Size')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        performance_plot_path = plots_dir / "performance_vs_size.png"
        plt.savefig(performance_plot_path)
        
        # Additional plots for each metric
        metrics_to_plot = {
            'precision': precision,
            'recall': recall,
            'mAP50': map50,
            'mAP50-95': map50_95
        }
        
        for metric_name, metric_values in metrics_to_plot.items():
            plt.figure(figsize=(8, 5))
            plt.plot(counts, metric_values, 'o-', color='royalblue')
            
            # Add trendline
            z = np.polyfit(np.log10(counts), metric_values, 1)
            p = np.poly1d(z)
            plt.plot(counts, p(np.log10(counts)), 'r--', alpha=0.7)
            
            plt.xscale('log')
            plt.xlabel('Number of Training Images (log scale)')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} vs. Training Set Size')
            plt.grid(True, alpha=0.3)
            
            # Save individual plot
            metric_plot_path = plots_dir / f"{metric_name}_vs_size.png"
            plt.savefig(metric_plot_path)
            
        logger.info(f"Saved performance plots to {plots_dir}")
        
    except ImportError:
        logger.warning("Matplotlib not available. Skipping plot generation.")
    except Exception as e:
        logger.exception(f"Error generating plots: {e}")

def run_experiment(args, logger):
    """Run the complete experiment pipeline."""
    start_time = time.time()
    logger.info(f"Starting experiment: {args.experiment_name}")
    
    # 1. Create experiment directory
    experiment_path = Path(args.output_path) / args.experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Initialize image selector
    selector = ImageSelector(
        args.data_yaml,
        args.output_path,
        args.experiment_name
    )
    
    # 3. Create incremental image sets
    logger.info("Creating incremental image sets")
    incremental_sets = selector.create_incremental_sets(
        num_sets=args.num_sets,
        max_images=args.max_images,
        scaling=args.scaling,
        seed=args.seed,
        strategy=args.strategy
    )
    
    # 4. Export image sets and create configs
    logger.info("Exporting image sets and creating configurations")
    output_configs = selector.export_image_sets(
        incremental_sets,
        args.pipeline_config,
        args.train_config
    )
    
    # Save experiment configuration
    experiment_config = {
        'name': args.experiment_name,
        'num_sets': args.num_sets,
        'max_images': args.max_images or len(selector.image_list),
        'scaling': args.scaling,
        'seed': args.seed,
        'strategy': args.strategy,
        'image_counts': sorted(list(incremental_sets.keys())),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(experiment_path / "experiment_config.json", 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    # 5. Process each dataset through the pipeline
    for count, config in sorted(output_configs.items()):
        logger.info(f"Processing dataset with {count} images")
        success = process_dataset(config['pipeline_config'], logger)
        
        if not success:
            logger.error(f"Failed to process dataset with {count} images")
            if args.continue_on_error:
                continue
            else:
                return False
    
    # 6. Train models for each processed dataset
    for count, config in sorted(output_configs.items()):
        logger.info(f"Training model with {count} images")
        
        # Check if the processed data exists
        if not config['processed_yaml_path'].exists():
            logger.error(f"Processed data not found for {count} images: {config['processed_yaml_path']}")
            if args.continue_on_error:
                continue
            else:
                return False
        
        success = train_model(config['train_config'], logger)
        
        if not success:
            logger.error(f"Failed to train model with {count} images")
            if args.continue_on_error:
                continue
            else:
                return False
    
    # 7. Collect and analyze results
    metrics = collect_metrics(experiment_path, logger)
    
    # 8. Generate plots
    generate_plots(metrics, experiment_path, logger)
    
    # Calculate experiment duration
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Experiment completed in {duration/3600:.2f} hours")
    
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run coral detection scaling experiment")
    parser.add_argument('--data_yaml', required=True, help='Path to the cgras_data.yaml file')
    parser.add_argument('--output_path', required=True, help='Base path for outputs')
    parser.add_argument('--pipeline_config', required=True, help='Path to pipeline config template')
    parser.add_argument('--train_config', required=True, help='Path to training config template')
    parser.add_argument('--experiment_name', default='CoralScaling', help='Name of the experiment')
    parser.add_argument('--num_sets', type=int, default=10, help='Number of different sized sets to create')
    parser.add_argument('--max_images', type=int, help='Maximum number of images to use')
    parser.add_argument('--scaling', choices=['log', 'linear'], default='log', help='Scaling of image counts')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--strategy', choices=['random', 'stratified_week', 'stratified_tile'], 
                       default='stratified_week', help='Selection strategy')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--continue_on_error', action='store_true', help='Continue experiment if a step fails')
    parser.add_argument('--skip_processing', action='store_true', help='Skip data processing step')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training step')
    parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat experiment with different seeds')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(
        log_level=logging.DEBUG if args.debug else logging.INFO,
        experiment_name=args.experiment_name
    )
    
    # Run experiment
    if args.repeat > 1:
        for i in range(args.repeat):
            # Use a different seed for each run if seed was provided
            if args.seed is not None:
                run_seed = args.seed + i
            else:
                run_seed = None
                
            run_name = f"{args.experiment_name}_run{i+1}"
            logger.info(f"Starting experiment run {i+1}/{args.repeat} with name '{run_name}'")
            
            # Create args for this run
            run_args = argparse.Namespace(
                **vars(args),
                experiment_name=run_name,
                seed=run_seed
            )
            
            success = run_experiment(run_args, logger)
            
            if not success and not args.continue_on_error:
                logger.error(f"Experiment run {i+1} failed. Stopping.")
                sys.exit(1)
    else:
        success = run_experiment(args, logger)
        
        if not success:
            logger.error("Experiment failed.")
            sys.exit(1)
    
    logger.info("All experiments completed successfully!")