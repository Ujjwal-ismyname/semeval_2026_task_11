import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

def get_model_and_tokenizer(model_name: str, use_lora: bool = True):
    """
    Loads model and tokenizer with 4-bit quantization and optional LoRA adapters.
    """
    print(f"Loading model: {model_name}")
    
    # Quantization Config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Llama usually has no pad token
    tokenizer.padding_side = "right" # Important for Causal LM usually, but for batch inference "left" is better. Trainer handles right padding usually.

    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for Training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    if use_lora:
        print("Applying LoRA adapters...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Target attention blocks usually
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    return model, tokenizer
