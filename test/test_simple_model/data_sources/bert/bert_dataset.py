

if __name__ == "__main__":
    dataset = BertSentimentDataset(split="train", max_samples=100)
    print(f"Dataset size: {len(dataset)}")

    input_ids, attention_mask, label = dataset[0]
    print(f"Input IDs shape: {len(input_ids)}")
    print(f"Attention mask shape: {len(attention_mask)}")
    print(f"Label: {label}")
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizer


class BertSentimentDataset(Dataset):
    def __init__(self, split="train", max_length=128, max_samples=None):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

        dataset = load_dataset("glue", "sst2", split=split)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.data = self._tokenize_dataset(dataset)

    def _tokenize_dataset(self, dataset):
        tokenized = []
        for item in dataset:
            encoding = self.tokenizer(
                item["sentence"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            tokenized.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "label": item["label"],
            })
        return tokenized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["input_ids"], item["attention_mask"], item["label"]
