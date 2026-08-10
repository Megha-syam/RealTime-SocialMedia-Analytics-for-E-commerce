import argparse
import json
from pathlib import Path
from random import Random
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.connectors import fetch_live_for_queries


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

POS_WORDS = {
    "excellent",
    "great",
    "good",
    "smooth",
    "premium",
    "fast",
    "improved",
    "reliable",
    "affordable",
}
NEG_WORDS = {
    "drain",
    "heat",
    "issue",
    "slow",
    "delay",
    "disappoint",
    "vibration",
    "stiff",
    "problem",
    "expensive",
}


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _weak_label_text(text: str) -> str:
    normalized = text.lower()
    pos = sum(1 for token in POS_WORDS if token in normalized)
    neg = sum(1 for token in NEG_WORDS if token in normalized)
    if pos == 0 and neg == 0:
        return "neutral"
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _pseudo_labels_with_teacher(texts):
    # Binary teacher; convert low-confidence predictions to neutral.
    from transformers import pipeline

    teacher = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    labels = []
    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        predictions = teacher(batch, truncation=True, max_length=256)
        for pred in predictions:
            score = float(pred.get("score", 0.0))
            label = str(pred.get("label", "")).lower()
            if score < 0.62:
                labels.append("neutral")
            elif "pos" in label:
                labels.append("positive")
            else:
                labels.append("negative")
    return labels


def _collect_live_training_samples(queries, limit_per_query, max_rows):
    rows = fetch_live_for_queries(queries, limit_per_query=limit_per_query)
    texts = []
    seen = set()
    for row in rows:
        text = (row.get("text") or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        texts.append(text)
        if len(texts) >= max_rows:
            break
    return texts


def _build_training_pairs(texts):
    if not texts:
        return [], []
    try:
        labels = _pseudo_labels_with_teacher(texts)
    except Exception:
        labels = [_weak_label_text(text) for text in texts]
    numeric_labels = [LABEL_TO_ID[label] for label in labels]
    return texts, numeric_labels


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == batch["labels"]).sum().item())
            total += int(batch["labels"].size(0))
    return (correct / total) if total else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Train local BERT sentiment model using real-time link data only."
    )
    parser.add_argument(
        "--queries",
        default="iPhone 15,Samsung S24,OnePlus 12,Bajaj CT 110 X ES",
        help="Comma-separated product queries.",
    )
    parser.add_argument("--limit-per-query", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    output_dir = BACKEND_ROOT / "models" / "bert_sentiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = _collect_live_training_samples(
        queries, limit_per_query=args.limit_per_query, max_rows=args.max_rows
    )
    texts, labels = _build_training_pairs(texts)

    if len(texts) < 40:
        raise RuntimeError(
            "Not enough live samples from links to train BERT. "
            "Use broader queries or retry later."
        )

    idx = list(range(len(texts)))
    Random(42).shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_ds = SentimentDataset(train_texts, train_labels, tokenizer)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for _ in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()

    accuracy = evaluate(model, val_loader, device)

    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "label_to_id": LABEL_TO_ID,
                "id_to_label": ID_TO_LABEL,
                "val_accuracy": accuracy,
                "live_samples": len(texts),
                "queries": queries,
            },
            handle,
            indent=2,
        )

    print(
        f"Saved local BERT model to {output_dir} "
        f"(val_accuracy={accuracy:.3f}, live_samples={len(texts)})"
    )


if __name__ == "__main__":
    main()
