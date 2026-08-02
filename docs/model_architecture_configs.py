"""Reference Hugging Face/PyTorch configs for the two from-scratch model families.

These configs reproduce the block-level dimensions documented in report.md.
The original E1/E2 checkpoints used project-specific implementations and
tokenizers, so this file is a readable configuration reference rather than a
claim that the published artifacts were instantiated through Transformers.
"""

from transformers import GPT2Config, LlamaConfig


def llama_style_60m_config() -> LlamaConfig:
    """E1: Llama-style decoder with RoPE, RMSNorm, GQA, and SwiGLU."""
    return LlamaConfig(
        vocab_size=12_000,
        hidden_size=768,
        intermediate_size=2_048,
        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=1_024,
        rope_theta=10_000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )


def gpt2_style_63m_config() -> GPT2Config:
    """E2: GPT-2-style decoder with learned positions, MHA, and GELU-new."""
    return GPT2Config(
        vocab_size=16_384,
        n_positions=1_024,
        n_ctx=1_024,
        n_embd=768,
        n_layer=7,
        n_head=12,
        n_inner=3_072,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=True,
    )


if __name__ == "__main__":
    print(llama_style_60m_config())
    print(gpt2_style_63m_config())
