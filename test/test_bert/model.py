import torch
from transformers import BertForSequenceClassification, BertConfig


def create_bert_model(
    vocab_size=30522,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
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

    batch_size, seq_length = 4, 32
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")

    labels = torch.randint(0, 2, (batch_size,))
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    print(f"Loss: {loss.item():.4f}")

    loss.backward()
    print("Backward pass completed successfully!")
