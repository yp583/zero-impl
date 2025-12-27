import torch
from transformers import BertForSequenceClassification, BertConfig


def create_bert_model(
    vocab_size=30522,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    max_position_embeddings=512,
    num_labels=2,
):
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_labels=num_labels,
    )
    return BertForSequenceClassification(config)


if __name__ == "__main__":
    model = create_bert_model()
    total_numel = sum([p.numel() for p in model.parameters()])
    print(total_numel)
