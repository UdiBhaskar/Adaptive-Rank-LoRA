"""
Adaptive Rank LoRA Training Script

This module implements a training script for fine-tuning large language models using 
Adaptive Rank LoRA (Low-Rank Adaptation). The script automatically analyzes the model's
weight matrices using spectral analysis to assign optimal ranks to different layers.

Key Features:
- Automatic rank assignment based on spectral properties
- Support for QLoRA, RSLoRA, and DoRA variants
- Integrated monitoring and logging
- Flexible configuration via dataclasses
- Support for various model architectures and quantization options

The script uses the TRL (Transformers Reinforcement Learning) library for training
with comprehensive support for PEFT (Parameter Efficient Fine-Tuning).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from trl import SFTConfig, SFTTrainer, TrlParser, get_kbit_device_map
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from utils import flatten_dict, setup_logging, BatchTimeCallback, load_raw_data
import torch
from dataclasses import asdict
import json
import os
from adaptive_rank_assignment import get_adaptive_lora_config


@dataclass
class LoraModelConfig:
    """
    Configuration class for LoRA model settings and quantization options.
    
    This class defines all the hyperparameters and settings related to the model
    loading, quantization, and LoRA configuration. It supports various LoRA
    variants including QLoRA, RSLoRA, and DoRA.
    
    Attributes:
        model_name_or_path: Path or name of the pre-trained model
        trust_remote_code: Whether to trust remote code when loading models
        attn_implementation: Attention implementation to use (flash_attention_2 recommended)
        use_peft: Whether to use Parameter Efficient Fine-Tuning
        use_qlora: Whether to use Quantized LoRA (4-bit or 8-bit)
        use_rslora: Whether to use Rank-Stabilized LoRA
        use_dora: Whether to use Weight-Decomposed Low-Rank Adaptation
        lora_r: Default LoRA rank (overridden by adaptive assignment)
        lora_alpha: Default LoRA alpha scaling parameter
        lora_dropout: Dropout rate for LoRA layers
        lora_target_modules: Target modules for LoRA adaptation
        lora_modules_to_save: Additional modules to keep unfrozen
        lora_task_type: Task type for LoRA configuration
        load_in_8bit/load_in_4bit: Quantization options
        bnb_4bit_quant_type: BitsAndBytes quantization type
        use_bnb_nested_quant: Whether to use nested quantization
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code when loading a model."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "help": (
                "Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=True,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )

    use_qlora: bool = field(
        default=False,
        metadata={"help": ("Whether to use Quantized LoRA")},
    )
    
    use_rslora: bool = field(
        default=False,
        metadata={"help": ("Whether to use RSLoRA")},
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable <a href='https://arxiv.org/abs/2402.09353'>'Weight-Decomposed Low-Rank Adaptation' (DoRA)</a>. This technique decomposes the updates of the "
                "weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
                "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
                "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger"
                "overhead than pure LoRA, so it is recommended to merge weights for inference."
            )
        },
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={
            "help": "The task_type to pass for LoRA (use SEQ_CLS for reward modeling)"
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": "use 8 bit precision for the base model - works only with LoRA"
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={
            "help": "use 4 bit precision for the base model - works only with LoRA"
        },
    )

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(
        default=False, metadata={"help": "use nested quantization"}
    )
    
    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if (
            isinstance(self.lora_target_modules, list)
            and len(self.lora_target_modules) == 1
        ):
            self.lora_target_modules = self.lora_target_modules[0]

@dataclass
class RankAssignmentConfig:
    """
    Configuration class for adaptive rank assignment parameters.
    
    This class controls how the spectral analysis results are used to assign
    LoRA ranks to different layers. The parameters control both the layer
    selection and rank assignment strategies.
    
    Attributes:
        top_n_percentile: Fraction of layers to select for LoRA adaptation (0.0-1.0).
            For example, 0.5 means only the top 50% most suitable layers will be adapted.
        min_rank: Minimum LoRA rank to assign to any selected layer.
        max_rank: Maximum LoRA rank to assign to any selected layer.
        rank_scaling: Method for interpolating ranks between min and max.
            Options: "linear" (uniform), "log" (logarithmic), "sqrt" (square root).
        alpha_factor: Multiplicative factor for computing LoRA alpha from rank.
            LoRA alpha = rank * alpha_factor. Higher values increase adaptation strength.
        w_hill: Weight for Hill alpha in the composite scoring function.
            Hill alpha measures the heavy-tailedness of eigenvalue distributions.
        w_mp_spikes: Weight for Marchenko-Pastur spikes in the composite scoring.
            MP spikes count eigenvalues above the random matrix bulk edge.
    
    Notes:
        - w_hill + w_mp_spikes should typically sum to 1.0 for interpretable scores
        - Higher w_hill emphasizes power law behavior in eigenvalue tails
        - Higher w_mp_spikes emphasizes structured (non-random) components
    """
    top_n_percentile: float = field(
        default=0.5, metadata={"help": "Fraction of layers to select for LoRA adaptation (0.0-1.0)"}
    )
    min_rank: int = field(
        default=4, metadata={"help": "Minimum LoRA rank to assign"}
    )
    max_rank: int = field(
        default=64, metadata={"help": "Maximum LoRA rank to assign"}
    )
    rank_scaling: str = field(
        default="linear", metadata={"help": "Rank scaling method: linear, log, or sqrt"}
    )
    alpha_factor: float = field(
        default=2, metadata={"help": "Multiplicative factor for LoRA alpha computation"}
    )
    w_hill: float = field(
        default=0.7, metadata={"help": "Weight for Hill alpha in composite score"}
    )
    w_mp_spikes: float = field(
        default=0.3, metadata={"help": "Weight for MP spikes in composite score"}
    )

@dataclass
class DataArguments:
    """
    Configuration class for dataset paths and data loading parameters.
    
    This class specifies the paths to training and validation datasets.
    The datasets should be in Parquet format for efficient loading.
    
    Attributes:
        train_dataset_path: Path to the training dataset file (Parquet format)
        val_dataset_path: Path to the validation dataset file (Parquet format)
    """
    train_dataset_path: str = field(
        default=None, metadata={"help": "Path to training dataset (Parquet format)"}
    )
    val_dataset_path: str = field(
        default=None, metadata={"help": "Path to validation dataset (Parquet format)"}
    )


def get_trainable_parameters(model):
    """
    Calculates the number of trainable parameters in a PyTorch model.
    
    This function is particularly useful for PEFT methods like LoRA, where
    only a small fraction of parameters are typically trainable.
    
    Args:
        model (torch.nn.Module): The PyTorch model to analyze
        
    Returns:
        tuple: A tuple containing:
            - trainable_params (int): Number of trainable parameters
            - all_param (int): Total number of parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, all_param


def create_datasets(tokenizer, data_args):
    """
    Creates training and validation datasets from the specified data paths.
    
    This function loads datasets in Parquet format and returns them as
    HuggingFace datasets ready for training.
    
    Args:
        tokenizer: The tokenizer (not used directly but maintains interface compatibility)
        data_args (DataArguments): Configuration containing dataset paths
        
    Returns:
        tuple: A tuple containing (train_dataset, validation_dataset)
    """
    dataset = load_raw_data(data_args)
    return dataset['train'], dataset['val']


def save_json(data, filename):
    """
    Saves data to a JSON file with error handling and pretty formatting.
    
    This utility function provides robust JSON saving with proper error
    handling and human-readable formatting.
    
    Args:
        data: The data structure to save (must be JSON serializable)
        filename (str): Path where the JSON file should be saved
        
    Raises:
        Exception: If the file cannot be written or data cannot be serialized
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)  # Pretty formatting with 4-space indentation
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Failed to save data to {filename}: {e}")


def main():
    """
    Main function for Adaptive Rank LoRA training.
    
    This function orchestrates the complete training pipeline:
    1. Parse configuration arguments
    2. Set up logging and save configuration
    3. Load tokenizer and model (with optional quantization)
    4. Perform spectral analysis for adaptive rank assignment
    5. Configure PEFT with adaptive ranks
    6. Create datasets and data collator
    7. Initialize and run the trainer
    8. Save the final model
    
    The function supports various LoRA variants (QLoRA, RSLoRA, DoRA) and
    automatically assigns optimal ranks to layers based on spectral analysis.
    """
    
    # Step 1: Parse command line arguments into configuration dataclasses
    parser = TrlParser((LoraModelConfig, SFTConfig, DataArguments, RankAssignmentConfig))
    loramodel_args, training_args, data_args, rank_assignment_args = parser.parse_args_into_dataclasses()

    # Step 2: Set up output directory and logging
    # Create a comprehensive configuration dictionary for logging and reproducibility
    args_data_dict = {
        "loramodel_args": asdict(loramodel_args),
        "training_args": asdict(training_args),
        "data_args": asdict(data_args),
        "rank_assignment_args": asdict(rank_assignment_args)
    }
    
    # Ensure output directory exists
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    
    # Set up logging to both file and console
    log_file_path = os.path.join(training_args.output_dir, training_args.run_name + ".log")
    logger = setup_logging(log_file_path)
    args_data_dict['data_args']['log_file_path'] = log_file_path

    # Save all training parameters for reproducibility
    training_params_path = os.path.join(training_args.output_dir, "training_params.json")
    save_json(args_data_dict, training_params_path)
    logger.info(f"Training parameters saved in {training_params_path}")
    
    # Step 3: Validate training configuration
    if training_args.group_by_length and training_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    # Set random seed for reproducibility
    set_seed(training_args.seed)
    
    # Step 4: Load tokenizer and configure chat template
    logger.info("Loading Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        loramodel_args.model_name_or_path, 
        trust_remote_code=loramodel_args.trust_remote_code
    )
    
    # Configure chat template for conversation formatting
    logger.info("Loading Chat Template ...")
    tokenizer.chat_template = open(
        "/data/function_calling/src_finetune/qwen_template.jinja"
    ).read()
    tokenizer.padding_side = "right"  # Required for proper attention mask handling
    
    # Step 5: Load model with optional quantization
    if loramodel_args.use_qlora:
        # QLoRA: Quantized LoRA for memory-efficient training
        logger.info("Loading BitsAndBytesConfig for QLoRA ...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=loramodel_args.load_in_8bit,
            load_in_4bit=loramodel_args.load_in_4bit,
            bnb_4bit_quant_type=loramodel_args.bnb_4bit_quant_type,  # NF4 or FP4
            bnb_4bit_use_double_quant=loramodel_args.use_bnb_nested_quant,  # Nested quantization
            bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype for better stability
        )

        logger.info("Loading Model with Quantization ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            loramodel_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map={"": Accelerator().local_process_index},  # Single GPU per process
            trust_remote_code=loramodel_args.trust_remote_code,
            attn_implementation="flash_attention_2",  # Memory-efficient attention
        )
    else:
        # Standard model loading without quantization
        logger.info("Loading Model without Quantization ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            loramodel_args.model_name_or_path,
            device_map=get_kbit_device_map(),  # Automatic device mapping
            torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
            trust_remote_code=loramodel_args.trust_remote_code,
            attn_implementation="flash_attention_2",  # Memory-efficient attention
        )
    
    # Step 6: Perform adaptive rank assignment using spectral analysis
    logger.info("Computing Adaptive LoRA Configuration ...")
    rank_pattern, alpha_pattern, target_regex = get_adaptive_lora_config(
        base_model,
        layer_selection_percentile=rank_assignment_args.top_n_percentile,
        minimum_rank=rank_assignment_args.min_rank,
        maximum_rank=rank_assignment_args.max_rank,
        rank_scaling_method=rank_assignment_args.rank_scaling,
        alpha_scaling_factor=rank_assignment_args.alpha_factor,
        hill_weight=rank_assignment_args.w_hill,
        mp_spikes_weight=rank_assignment_args.w_mp_spikes
    )
    
    # Step 7: Configure PEFT (Parameter Efficient Fine-Tuning)
    if loramodel_args.use_peft:
        logger.info("Configuring Adaptive LoRA PEFT ...")
        
        # Create LoRA configuration with adaptive ranks
        peft_config = LoraConfig(
            task_type=loramodel_args.lora_task_type,  # CAUSAL_LM for language modeling
            r=loramodel_args.lora_r,  # Default rank (overridden by rank_pattern)
            use_rslora=loramodel_args.use_rslora,  # Rank-Stabilized LoRA
            target_modules=target_regex,  # Regex pattern for target modules
            lora_alpha=loramodel_args.lora_alpha,  # Default alpha (overridden by alpha_pattern)
            lora_dropout=loramodel_args.lora_dropout,  # Dropout for regularization
            use_dora=loramodel_args.use_dora,  # Weight-Decomposed Low-Rank Adaptation
            bias="none",  # Don't adapt bias parameters
            modules_to_save=loramodel_args.lora_modules_to_save,  # Additional modules to save
            rank_pattern=rank_pattern,  # Adaptive rank assignments per layer
            alpha_pattern=alpha_pattern,  # Adaptive alpha values per layer
            # init_lora_weights="pissa"  # Alternative initialization method
        )
        
        # Log the adaptive configuration for transparency
        logger.info(f"Adaptive ranks assigned to {len(rank_pattern)} layers")
        logger.info(f"Rank range: {min(rank_pattern.values())} - {max(rank_pattern.values())}")
        logger.info(f"Alpha range: {min(alpha_pattern.values())} - {max(alpha_pattern.values())}")
    else:
        peft_config = None
        logger.info("PEFT disabled - using full fine-tuning")
    

    # Log model configuration for debugging
    logger.info(f"Attention implementation: {base_model.config._attn_implementation}")
    
    # Step 8: Load and prepare datasets
    logger.info("Loading Datasets ...")
    train_dataset, valid_dataset = create_datasets(tokenizer, data_args)
    
    # Create data collator for language modeling (causal LM)
    # MLM=False indicates causal (autoregressive) language modeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Step 9: Initialize the SFT (Supervised Fine-Tuning) trainer
    logger.info("Instantiating SFTTrainer ...")
    trainer = SFTTrainer(
        model=base_model,                           # Base model to fine-tune
        train_dataset=train_dataset,                # Training data
        eval_dataset=valid_dataset,                 # Validation data
        data_collator=collator,                     # Data collation strategy
        peft_config=peft_config,                    # PEFT configuration (LoRA)
        tokenizer=tokenizer,                        # Tokenizer
        args=training_args,                         # Training arguments
        callbacks=[BatchTimeCallback(logger)]      # Custom callbacks for monitoring
    )

    # Step 10: Log parameter efficiency statistics
    trainable_params, all_param = get_trainable_parameters(trainer.model)
    efficiency_percentage = 100 * trainable_params / all_param
    
    logger.info(
        f"Parameter efficiency: {trainable_params:,} trainable / {all_param:,} total "
        f"({efficiency_percentage:.3f}% trainable)"
    )
    
    # Step 11: Execute training
    logger.info("Starting Training ...")
    trainer.train()

    # Step 12: Save the final trained model
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.save_model(output_dir)
    # Note: Tokenizer is typically saved with the model automatically
    logger.info(f"Final checkpoint saved in {output_dir}")
    
    # Step 13: Clean up GPU memory
    torch.cuda.empty_cache()
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()