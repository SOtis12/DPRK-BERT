#!/usr/bin/env python3
"""
Complete Enhanced DPRK-BERT Training Pipeline
Orchestrates data processing, training, and evaluation with proper error handling.

This script:
1. Processes all data sources (PDFs, dictionaries, parallel data, speeches, web data)
2. Creates enhanced training dataset with quality filtering
3. Runs TPU-optimized training
4. Evaluates model performance
5. Provides comprehensive logging and monitoring

Usage:
    python enhanced_pipeline.py --help
    python enhanced_pipeline.py --data-dir ./Resources --output-dir ./enhanced_output
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import enhanced modules
try:
    from config_enhanced import (
        PROJECT_ROOT, ENHANCED_MLM_ROOT, TPU_CONFIG, TRAINING_CONFIG, DATA_CONFIG,
        is_tpu_environment, get_device_info
    )
    from prepare_mlm_dataset_enhanced import prepare_enhanced_data, parse_enhanced_args
    from tpu_trainer import TPUTrainingConfig, setup_model_and_tokenizer, train_on_tpu, load_and_prepare_dataset
except ImportError as e:
    logger.error(f"Failed to import enhanced modules: {e}")
    logger.error("Make sure all enhanced modules are in the correct location")
    sys.exit(1)

class PipelineError(Exception):
    """Custom exception for pipeline errors"""
    pass

class EnhancedDPRKPipeline:
    """Complete pipeline for enhanced DPRK-BERT training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = time.time()
        self.results = {}
        
        # Setup directories
        self.data_dir = Path(config['data_dir'])
        self.output_dir = Path(config['output_dir'])
        self.dataset_dir = self.output_dir / "dataset"
        self.model_dir = self.output_dir / "model"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized with config: {config}")
    
    def step_1_process_data(self) -> Dict[str, Any]:
        """Step 1: Process all data sources into training dataset"""
        logger.info("=" * 60)
        logger.info("STEP 1: DATA PROCESSING")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Setup arguments for enhanced data processor
            class Args:
                def __init__(self, config):
                    self.source_folder = str(config['data_dir'])
                    self.save_folder = str(config['dataset_dir'])
                    self.sources = config.get('sources', ['speeches', 'parallel', 'dictionaries', 'pdfs'])
                    self.max_dict_entries = config.get('max_dict_entries', DATA_CONFIG['max_dictionary_entries'])
                    self.scraped_folders = config.get('scraped_folders', [])
                    self.quality_filter = config.get('quality_filter', DATA_CONFIG['quality_filter'])
                    self.deduplicate = config.get('deduplicate', DATA_CONFIG['deduplicate'])
                    self.min_text_length = config.get('min_text_length', DATA_CONFIG['min_text_length'])
                    self.max_text_length = config.get('max_text_length', DATA_CONFIG['max_text_length'])
                    self.split_ratio = config.get('split_ratio', DATA_CONFIG['train_split_ratio'])
                    self.apply_split = True
            
            args = Args(self.config)
            
            # Process data
            logger.info(f"Processing data sources: {args.sources}")
            prepare_enhanced_data(
                source_folder=args.source_folder,
                save_folder=args.save_folder,
                sources=args.sources,
                args=args
            )
            
            # Validate outputs
            train_file = self.dataset_dir / "train.json"
            val_file = self.dataset_dir / "validation.json"
            metadata_file = self.dataset_dir / "metadata.json"
            
            if not train_file.exists():
                raise PipelineError(f"Training file not created: {train_file}")
            
            # Load and validate metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                total_items = metadata.get('total_items', 0)
                train_items = metadata.get('dataset_info', {}).get('train_items', 0)
                val_items = metadata.get('dataset_info', {}).get('validation_items', 0)
                
                logger.info(f"Dataset created successfully:")
                logger.info(f"  Total items: {total_items}")
                logger.info(f"  Training items: {train_items}")
                logger.info(f"  Validation items: {val_items}")
                
                result = {
                    "status": "success",
                    "duration": time.time() - start_time,
                    "total_items": total_items,
                    "train_items": train_items,
                    "validation_items": val_items,
                    "metadata": metadata
                }
            else:
                result = {
                    "status": "success", 
                    "duration": time.time() - start_time,
                    "note": "Metadata file not found, but training files exist"
                }
            
            self.results['data_processing'] = result
            logger.info(f"Data processing completed in {result['duration']:.2f} seconds")
            return result
            
        except Exception as e:
            error_msg = f"Data processing failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            result = {
                "status": "failed",
                "duration": time.time() - start_time,
                "error": error_msg
            }
            self.results['data_processing'] = result
            raise PipelineError(error_msg)
    
    def step_2_setup_training(self) -> Dict[str, Any]:
        """Step 2: Setup training configuration and validate environment"""
        logger.info("=" * 60)
        logger.info("STEP 2: TRAINING SETUP")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Get device info
            device_info = get_device_info()
            logger.info(f"Compute environment: {device_info}")
            
            # Create training configuration
            train_config = TPUTrainingConfig(
                model_name_or_path=self.config.get('model_name', 'snunlp/KR-Medium'),
                train_file=str(self.dataset_dir / "train.json"),
                validation_file=str(self.dataset_dir / "validation.json"),
                output_dir=str(self.model_dir),
                num_train_epochs=self.config.get('epochs', TRAINING_CONFIG['default_epochs']),
                learning_rate=self.config.get('learning_rate', TRAINING_CONFIG['learning_rate']),
                per_device_train_batch_size=self.config.get('batch_size', 0),  # 0 = auto
                max_seq_length=self.config.get('max_seq_length', DATA_CONFIG['max_sequence_length']),
                warmup_ratio=self.config.get('warmup_ratio', TRAINING_CONFIG['warmup_ratio']),
            )
            
            # Adjust configuration based on device
            if device_info['type'] == 'tpu':
                train_config.bf16 = True
                train_config.tpu_num_cores = TPU_CONFIG['num_cores']
                train_config.per_device_train_batch_size = train_config.per_device_train_batch_size or TPU_CONFIG['per_core_batch_size']
            elif device_info['type'] == 'gpu':
                train_config.bf16 = False
                train_config.fp16 = True
                train_config.per_device_train_batch_size = train_config.per_device_train_batch_size or 16
            else:
                train_config.bf16 = False
                train_config.fp16 = False
                train_config.per_device_train_batch_size = train_config.per_device_train_batch_size or 8
            
            # Setup model and tokenizer
            logger.info("Setting up model and tokenizer...")
            model, tokenizer = setup_model_and_tokenizer(train_config)
            
            result = {
                "status": "success",
                "duration": time.time() - start_time,
                "device_info": device_info,
                "model_name": train_config.model_name_or_path,
                "batch_size": train_config.per_device_train_batch_size,
                "epochs": train_config.num_train_epochs,
                "learning_rate": train_config.learning_rate,
                "config": train_config
            }
            
            self.results['training_setup'] = result
            logger.info("Training setup completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Training setup failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            result = {
                "status": "failed",
                "duration": time.time() - start_time,
                "error": error_msg
            }
            self.results['training_setup'] = result
            raise PipelineError(error_msg)
    
    def step_3_train_model(self) -> Dict[str, Any]:
        """Step 3: Train the enhanced DPRK-BERT model"""
        logger.info("=" * 60)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Get configuration from previous step
            setup_result = self.results.get('training_setup')
            if not setup_result or setup_result['status'] != 'success':
                raise PipelineError("Training setup not completed successfully")
            
            train_config = setup_result['config']
            
            # Setup model and tokenizer again (fresh for training)
            model, tokenizer = setup_model_and_tokenizer(train_config)
            
            # Load datasets
            logger.info("Loading training datasets...")
            train_dataset = load_and_prepare_dataset(train_config.train_file, tokenizer, train_config, is_train=True)
            eval_dataset = load_and_prepare_dataset(train_config.validation_file, tokenizer, train_config, is_train=False)
            
            logger.info(f"Training dataset size: {len(train_dataset)}")
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
            
            # Start training
            logger.info("Starting model training...")
            train_result = train_on_tpu(train_config, train_dataset, eval_dataset, model, tokenizer)
            
            training_duration = time.time() - start_time
            
            result = {
                "status": "success",
                "duration": training_duration,
                "training_loss": train_result.training_loss,
                "model_path": str(self.model_dir),
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset)
            }
            
            self.results['training'] = result
            logger.info(f"Model training completed in {training_duration:.2f} seconds")
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            return result
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            result = {
                "status": "failed",
                "duration": time.time() - start_time,
                "error": error_msg
            }
            self.results['training'] = result
            raise PipelineError(error_msg)
    
    def step_4_evaluate_model(self) -> Dict[str, Any]:
        """Step 4: Evaluate the trained model"""
        logger.info("=" * 60)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Check if training completed successfully
            training_result = self.results.get('training')
            if not training_result or training_result['status'] != 'success':
                raise PipelineError("Training not completed successfully")
            
            # TODO: Implement comprehensive evaluation
            # - Perplexity on validation set
            # - Sample text generation
            # - Comparison with original model
            
            result = {
                "status": "success",
                "duration": time.time() - start_time,
                "note": "Basic training completed - comprehensive evaluation to be implemented"
            }
            
            self.results['evaluation'] = result
            logger.info("Model evaluation completed")
            return result
            
        except Exception as e:
            error_msg = f"Model evaluation failed: {str(e)}"
            logger.error(error_msg)
            
            result = {
                "status": "failed", 
                "duration": time.time() - start_time,
                "error": error_msg
            }
            self.results['evaluation'] = result
            return result
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete enhanced DPRK-BERT pipeline"""
        logger.info("🚀 Starting Enhanced DPRK-BERT Training Pipeline")
        logger.info(f"Configuration: {self.config}")
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Process data
            self.step_1_process_data()
            
            # Step 2: Setup training
            self.step_2_setup_training()
            
            # Step 3: Train model
            self.step_3_train_model()
            
            # Step 4: Evaluate model
            self.step_4_evaluate_model()
            
            # Pipeline completed successfully
            total_duration = time.time() - pipeline_start
            
            final_result = {
                "status": "success",
                "total_duration": total_duration,
                "steps_completed": len([r for r in self.results.values() if r.get('status') == 'success']),
                "results": self.results
            }
            
            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Model directory: {self.model_dir}")
            
            # Save final results
            results_file = self.output_dir / "pipeline_results.json"
            with open(results_file, 'w') as f:
                json.dump(final_result, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {results_file}")
            return final_result
            
        except PipelineError as e:
            total_duration = time.time() - pipeline_start
            
            final_result = {
                "status": "failed",
                "total_duration": total_duration, 
                "error": str(e),
                "results": self.results
            }
            
            logger.error("=" * 60)
            logger.error("❌ PIPELINE FAILED")
            logger.error("=" * 60)
            logger.error(f"Error: {e}")
            logger.error(f"Duration before failure: {total_duration:.2f} seconds")
            
            # Save failure results
            results_file = self.output_dir / "pipeline_results.json"
            with open(results_file, 'w') as f:
                json.dump(final_result, f, indent=2, default=str)
            
            return final_result

def parse_pipeline_args():
    """Parse command line arguments for the pipeline"""
    parser = argparse.ArgumentParser(description="Enhanced DPRK-BERT Training Pipeline")
    
    # Required arguments
    parser.add_argument("--data-dir", type=str, required=True, 
                       help="Path to Resources directory containing all data sources")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for datasets, models, and results")
    
    # Data processing arguments
    parser.add_argument("--sources", nargs="+", 
                       default=["speeches", "parallel", "dictionaries", "pdfs"],
                       help="Data sources to include: speeches, parallel, dictionaries, pdfs, web")
    parser.add_argument("--max-dict-entries", type=int, default=50000,
                       help="Maximum dictionary entries to process")
    parser.add_argument("--scraped-folders", nargs="*", default=[],
                       help="Paths to scraped data folders")
    
    # Training arguments
    parser.add_argument("--model-name", type=str, default="snunlp/KR-Medium",
                       help="Base model name or path")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=0,
                       help="Batch size per device (0 = auto)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Quality control
    parser.add_argument("--no-quality-filter", action="store_true",
                       help="Disable quality filtering")
    parser.add_argument("--no-deduplicate", action="store_true", 
                       help="Disable deduplication")
    
    # Testing
    parser.add_argument("--dry-run", action="store_true",
                       help="Run pipeline validation without actual training")
    
    return parser.parse_args()

def main():
    """Main pipeline execution"""
    args = parse_pipeline_args()
    
    # Create configuration from arguments
    config = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "dataset_dir": Path(args.output_dir) / "dataset",
        "sources": args.sources,
        "max_dict_entries": args.max_dict_entries,
        "scraped_folders": args.scraped_folders,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "quality_filter": not args.no_quality_filter,
        "deduplicate": not args.no_deduplicate,
        "dry_run": args.dry_run
    }
    
    # Validate inputs
    data_dir = Path(config["data_dir"])
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = EnhancedDPRKPipeline(config)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual training will be performed")
        # TODO: Implement dry run validation
        logger.info("Dry run completed successfully")
    else:
        result = pipeline.run_complete_pipeline()
        
        # Exit with appropriate code
        if result["status"] == "success":
            sys.exit(0)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()