from test_tinyuniverse.transformers_main.src.transformers.models.llama import LlamaConfig, LlamaModel
import torch


def run_llama():
    llamaconfig = LlamaConfig(
        vacab_size=32000,
        hidden_size=2048,
        intermediate_size=11000//2,
        num_hidden_layers=16,
        num_attention_heads=16,
        max_position_embeddings=1024,
        _attn_implementation='eager'
    )

    llamamodel = LlamaModel(config=llamaconfig)

    input_ids = torch.randint(0, llamaconfig.vocab_size, (4, 30))

    res = llamamodel(input_ids)

    print(res)


if __name__ == '__main__':
    run_llama()







