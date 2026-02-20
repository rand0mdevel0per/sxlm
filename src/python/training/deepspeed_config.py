"""DeepSpeed ZeRO-3 configuration for distributed training"""

def get_deepspeed_config(hidden_dim: int = 4096) -> dict:
    """Get DeepSpeed ZeRO-3 configuration"""
    return {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 4,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": hidden_dim * hidden_dim,
            "stage3_prefetch_bucket_size": hidden_dim * hidden_dim,
            "stage3_param_persistence_threshold": hidden_dim * 10
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    }
