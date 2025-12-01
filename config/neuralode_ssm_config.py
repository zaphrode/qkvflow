"""
Configuration for Neural ODE + SSM Language Models
"""

from dataclasses import dataclass
from typing import Optional
from levanter.models.gpt2 import Gpt2Config


@dataclass
class NeuralOdeSSMConfig:
    """Configuration for Neural ODE Transformer with SSM replacing FFN"""
    
    # Base transformer config
    gpt2_config: Gpt2Config
    
    # Vocabulary size (GPT-2 default: 50257)
    vocab_size: int = 50257
    
    # Time embedding dimensions
    time_embedding_dim: int = 64
    sinusoidal_dim: int = 32
    
    # SSM-specific parameters
    ssm_state_size: int = 64  # Dimensionality of SSM hidden state
    
    # ODE solver settings  
    use_adaptive_solver: bool = False  # Whether to use adaptive ODE solver
    num_ode_steps: int = 1  # For fixed-step solvers

    @staticmethod
    def small_ssm() -> "NeuralOdeSSMConfig":
        """Small model for quick experiments (~19M params)"""
        return NeuralOdeSSMConfig(
            gpt2_config=Gpt2Config(
                seq_len=512,
                hidden_dim=256,
                num_heads=4,
                num_layers=6,
                gradient_checkpointing=False,
                resid_pdrop=0.1,
                embed_pdrop=0.1,
                attn_pdrop=0.1,
            ),
            time_embedding_dim=64,
            sinusoidal_dim=32,
            ssm_state_size=64,
        )

    @staticmethod
    def medium_ssm() -> "NeuralOdeSSMConfig":
        """Medium model (~100M params)"""
        return NeuralOdeSSMConfig(
            gpt2_config=Gpt2Config(
                seq_len=1024,
                hidden_dim=512,
                num_heads=8,
                num_layers=12,
                gradient_checkpointing=True,
                resid_pdrop=0.1,
                embed_pdrop=0.1,
                attn_pdrop=0.1,
            ),
            time_embedding_dim=128,
            sinusoidal_dim=64,
            ssm_state_size=128,
        )

    @staticmethod
    def large_ssm() -> "NeuralOdeSSMConfig":
        """Large model (~350M params)"""
        return NeuralOdeSSMConfig(
            gpt2_config=Gpt2Config(
                seq_len=2048,
                hidden_dim=1024,
                num_heads=16,
                num_layers=24,
                gradient_checkpointing=True,
                resid_pdrop=0.1,
                embed_pdrop=0.1,
                attn_pdrop=0.1,
            ),
            time_embedding_dim=256,
            sinusoidal_dim=128,
            ssm_state_size=256,
        )

    @staticmethod
    def from_dict(config_dict: dict) -> "NeuralOdeSSMConfig":
        """Create config from dictionary"""
        gpt2_dict = {
            k: v for k, v in config_dict.items()
            if k in Gpt2Config.__dataclass_fields__
        }
        gpt2_config = Gpt2Config(**gpt2_dict)

        ssm_dict = {
            k: v for k, v in config_dict.items()
            if k in NeuralOdeSSMConfig.__dataclass_fields__ and k != 'gpt2_config'
        }

        return NeuralOdeSSMConfig(gpt2_config=gpt2_config, **ssm_dict)


# Preset configs for ablations
SSM_ABLATION_CONFIGS = {
    "ssm_state_32": {"ssm_state_size": 32},
    "ssm_state_64": {"ssm_state_size": 64},
    "ssm_state_128": {"ssm_state_size": 128},
    "ssm_state_256": {"ssm_state_size": 256},
}
