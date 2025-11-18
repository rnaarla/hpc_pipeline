#!/usr/bin/env python3
"""
Config-driven CLI Orchestrator for HPC Pipeline
"""

import os
import sys
import time
import json
import yaml
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import torch.distributed as dist

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("Orchestrator")

@dataclass
class JobMetadata:
    """Job metadata for lineage tracking."""
    job_id: str
    start_time: str
    config_hash: str
    git_commit: Optional[str]
    environment: Dict[str, str]
    slurm_job_id: Optional[str] = None
    k8s_job_name: Optional[str] = None

@dataclass
class PipelineConfig:
    """Pipeline configuration schema."""
    # Core components
    profiling: Dict[str, Any]
    training: Dict[str, Any]
    data: Dict[str, Any]
    observability: Dict[str, Any]
    
    # Infrastructure
    infrastructure: Dict[str, Any]
    distributed: Dict[str, Any]
    
    # Execution
    execution: Dict[str, Any]
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

class ComponentManager:
    """Manages individual pipeline components."""
    
    def __init__(self, config: PipelineConfig, metadata: JobMetadata):
        self.config = config
        self.metadata = metadata
        self.components_status = {}
    
    def run_profiling(self) -> bool:
        """Run profiling component."""
        logger.info("Starting profiling component...")
        
        try:
            cmd = [
                sys.executable, "-m", "profiling.pytorch_profiler",
                "--config", str(self.config.profiling.get('config_file', 'configs/profiling.yaml')),
                "--trace-steps", str(self.config.profiling.get('trace_steps', 10))
            ]
            
            if self.config.profiling.get('measure_overhead', False):
                cmd.append("--measure-overhead")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info("Profiling completed successfully")
                self.components_status['profiling'] = 'success'
                return True
            else:
                logger.error(f"Profiling failed: {result.stderr}")
                self.components_status['profiling'] = 'failed'
                return False
                
        except Exception as e:
            logger.error(f"Profiling component error: {e}")
            self.components_status['profiling'] = 'error'
            return False
    
    def run_training(self) -> bool:
        """Run training component."""
        logger.info("Starting training component...")
        
        try:
            # Determine training type
            training_type = self.config.training.get('type', 'amp')
            
            if training_type == 'amp':
                cmd = [sys.executable, "-m", "optimization.amp_training"]
            elif training_type == 'ddp':
                # Use torchrun for distributed training
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={self.config.distributed.get('gpus_per_node', 1)}",
                    f"--nnodes={self.config.distributed.get('num_nodes', 1)}",
                    "-m", "distributed.ddp_training"
                ]
            else:
                raise ValueError(f"Unknown training type: {training_type}")
            
            # Add config file
            config_file = self.config.training.get('config_file', 'configs/training.yaml')
            cmd.extend(["--config", str(config_file)])
            
            # Add checkpoint resume if specified
            if self.config.training.get('resume_checkpoint'):
                cmd.extend(["--checkpoint", str(self.config.training['resume_checkpoint'])])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                logger.info("Training completed successfully")
                self.components_status['training'] = 'success'
                return True
            else:
                logger.error(f"Training failed: {result.stderr}")
                self.components_status['training'] = 'failed'
                return False
                
        except Exception as e:
            logger.error(f"Training component error: {e}")
            self.components_status['training'] = 'error'
            return False
    
    def run_data_pipeline(self) -> bool:
        """Run data pipeline component."""
        logger.info("Starting data pipeline component...")
        # Implementation would go here
        self.components_status['data'] = 'success'
        return True
    
    def run_observability(self) -> bool:
        """Run observability component."""
        logger.info("Starting observability component...")
        # Implementation would go here
        self.components_status['observability'] = 'success'
        return True
    
    def run_benchmarking(self) -> bool:
        """Run benchmarking component."""
        logger.info("Starting benchmarking component...")
        
        try:
            cmd = [
                sys.executable, "-m", "benchmarking.roofline_analysis",
                "--model-size", str(self.config.training.get('model_size', '1B'))
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                logger.info("Benchmarking completed successfully")
                self.components_status['benchmarking'] = 'success'
                return True
            else:
                logger.error(f"Benchmarking failed: {result.stderr}")
                self.components_status['benchmarking'] = 'failed'
                return False
                
        except Exception as e:
            logger.error(f"Benchmarking component error: {e}")
            self.components_status['benchmarking'] = 'error'
            return False

class HPCOrchestrator:
    """Main orchestrator for HPC pipeline."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = PipelineConfig.from_file(config_path)
        self.metadata = self._create_job_metadata()
        self.component_manager = ComponentManager(self.config, self.metadata)
        
        # Create output directory
        self.output_dir = Path(self.config.execution.get('output_dir', './outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
    def _create_job_metadata(self) -> JobMetadata:
        """Create job metadata for lineage tracking."""
        import hashlib
        
        # Calculate config hash
        config_str = yaml.dump(asdict(self.config), sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        
        # Get git commit if available
        git_commit = None
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                git_commit = result.stdout.strip()[:8]
        except:
            pass
        
        # Generate job ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"hpc_job_{timestamp}_{config_hash}"
        
        return JobMetadata(
            job_id=job_id,
            start_time=datetime.now().isoformat(),
            config_hash=config_hash,
            git_commit=git_commit,
            environment=dict(os.environ),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            k8s_job_name=os.environ.get('JOB_NAME')
        )
    
    def _save_metadata(self):
        """Save job metadata to file."""
        metadata_file = self.output_dir / f"{self.metadata.job_id}_metadata.json"
        
        metadata_dict = asdict(self.metadata)
        metadata_dict['components_status'] = self.component_manager.components_status
        metadata_dict['config'] = asdict(self.config)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Job metadata saved: {metadata_file}")
    
    def _detect_environment(self) -> str:
        """Detect execution environment (SLURM, K8s, local)."""
        if os.environ.get('SLURM_JOB_ID'):
            return 'slurm'
        elif os.environ.get('KUBERNETES_SERVICE_HOST'):
            return 'kubernetes'
        else:
            return 'local'
    
    def _setup_distributed(self):
        """Setup distributed training if needed."""
        if self.config.distributed.get('enabled', False):
            # Set environment variables for distributed training
            if 'RANK' not in os.environ:
                os.environ['RANK'] = str(os.environ.get('LOCAL_RANK', '0'))
            if 'WORLD_SIZE' not in os.environ:
                os.environ['WORLD_SIZE'] = str(self.config.distributed.get('world_size', '1'))
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = self.config.distributed.get('master_addr', 'localhost')
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = str(self.config.distributed.get('master_port', '12355'))
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        logger.info(f"Starting HPC pipeline job: {self.metadata.job_id}")
        logger.info(f"Environment: {self._detect_environment()}")
        logger.info(f"Config hash: {self.metadata.config_hash}")
        
        # Setup distributed if needed
        self._setup_distributed()
        
        # Get execution order
        execution_order = self.config.execution.get('order', [
            'profiling', 'training', 'data', 'observability', 'benchmarking'
        ])
        
        success = True
        start_time = time.time()
        
        try:
            for component in execution_order:
                logger.info(f"Executing component: {component}")
                
                if component == 'profiling':
                    if not self.component_manager.run_profiling():
                        success = False
                        break
                elif component == 'training':
                    if not self.component_manager.run_training():
                        success = False
                        break
                elif component == 'data':
                    if not self.component_manager.run_data_pipeline():
                        success = False
                        break
                elif component == 'observability':
                    if not self.component_manager.run_observability():
                        success = False
                        break
                elif component == 'benchmarking':
                    if not self.component_manager.run_benchmarking():
                        success = False
                        break
                else:
                    logger.warning(f"Unknown component: {component}")
            
            execution_time = time.time() - start_time
            
            if success:
                logger.info(f"Pipeline completed successfully in {execution_time:.2f}s")
            else:
                logger.error(f"Pipeline failed after {execution_time:.2f}s")
            
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user")
            success = False
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            success = False
        finally:
            # Always save metadata
            self._save_metadata()
        
        return success


def create_dummy_config() -> Path:
    """Create a dummy configuration for testing."""
    config = {
        'profiling': {
            'config_file': 'configs/profiling.yaml',
            'trace_steps': 5,
            'measure_overhead': True
        },
        'training': {
            'type': 'amp',
            'config_file': 'configs/training.yaml',
            'model_size': '1B'
        },
        'data': {
            'dataset_path': '/data/training',
            'batch_size': 32
        },
        'observability': {
            'prometheus_enabled': True,
            'grafana_enabled': True
        },
        'infrastructure': {
            'backend': 'local'
        },
        'distributed': {
            'enabled': False,
            'world_size': 1
        },
        'execution': {
            'output_dir': './outputs',
            'order': ['profiling', 'training', 'benchmarking']
        }
    }
    
    config_file = Path('configs/dummy_pipeline.yaml')
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_file


def main():
    parser = argparse.ArgumentParser(description="HPC Pipeline Orchestrator")
    parser.add_argument("--config", type=str, required=True, 
                       help="Pipeline configuration file")
    parser.add_argument("--create-dummy", action="store_true",
                       help="Create dummy configuration")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate configuration")
    args = parser.parse_args()
    
    if args.create_dummy:
        dummy_config = create_dummy_config()
        logger.info(f"Dummy configuration created: {dummy_config}")
        return
    
    # Load and validate configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        orchestrator = HPCOrchestrator(config_path)
        
        if args.validate_only:
            logger.info("Configuration validation successful")
            return
        
        # Run pipeline
        success = orchestrator.run_pipeline()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
