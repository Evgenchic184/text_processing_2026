"""Shared helpers for Laboratory work #4: text classification of tweets.

The module contains data loading/splitting utilities, tokenization, vocabulary
helpers, a small bidirectional RNN classifier, training/evaluation routines and
optional BERT fine-tuning helpers.  The BERT helpers import ``transformers``
lazily so the rest of the module works in environments where it is not
installed.
"""

from __future__ import annotations

import importlib.util
import random
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.utils import stop_words

SEED = 42
LABELS = [
    "age",
    "ethnicity",
    "gender",
    "not_cyberbullying",
    "other_cyberbullying",
    "religion",
]
TEXT_COL = "text"
LABEL_COL = "cyberbullying_type"
DEFAULT_DATA_PATH = "data/tweets.csv"

NEGATIONS = {
    "not",
    "no",
    "nor",
    "never",
    "none",
    "nobody",
    "nothing",
    "neither",
    "nowhere",
    "cannot",
}

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_RT_RE = re.compile(r"\brt\b")
_TOKEN_RE = re.compile(r"[a-z']+")

_APOSTROPHELESS_NEGATIONS = {
    "dont",
    "doesnt",
    "didnt",
    "isnt",
    "wasnt",
    "werent",
    "wont",
    "cant",
    "couldnt",
    "shouldnt",
    "wouldnt",
    "havent",
    "hasnt",
    "hadnt",
    "aint",
    "mustnt",
    "mightnt",
    "neednt",
}

_TRANSFORMERS_ERROR = (
    "The 'transformers' package is required for BERT experiments. "
    "Install it with: uv pip install transformers"
)


def _normalize_negation(token: str) -> str:
    """Normalize negation contractions to the token ``not``."""
    if token.endswith("n't") or token in _APOSTROPHELESS_NEGATIONS:
        return "not"
    return token


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and mps_backend.is_available()
        and hasattr(torch, "mps")
    ):
        torch.mps.manual_seed(seed)


def load_tweets(
    path: str = DEFAULT_DATA_PATH,
    per_class: Optional[int] = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Load and deduplicate the tweets dataset.

    Parameters
    ----------
    path:
        Path to the CSV file containing ``tweet_text`` and
        ``cyberbullying_type`` columns.
    per_class:
        If given, sample at most this many rows from every class.  ``None``
        keeps the full deduplicated dataset.
    seed:
        Random seed used for class-wise sampling.

    Returns
    -------
    pandas.DataFrame
        A frame with ``text`` and ``cyberbullying_type`` columns and a clean
        integer index.
    """
    df = pd.read_csv(path)
    if "tweet_text" in df.columns and TEXT_COL not in df.columns:
        df = df.rename(columns={"tweet_text": TEXT_COL})

    missing = [col for col in (TEXT_COL, LABEL_COL) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path!r}: {missing}")

    df = df[[TEXT_COL, LABEL_COL]].dropna(subset=[TEXT_COL, LABEL_COL])
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

    if per_class is not None:
        n_per_class = max(int(per_class), 0)
        chunks = [
            group.sample(n=min(len(group), n_per_class), random_state=seed)
            for _, group in df.groupby(LABEL_COL, sort=False)
        ]
        df = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else df.iloc[0:0].copy()
        )

    return df.reset_index(drop=True)


def _split_once(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` once, falling back to a non-stratified split if needed."""
    if test_size <= 0 or len(df) < 2:
        return df.copy(), df.iloc[0:0].copy()

    counts = df[label_col].value_counts()
    stratify = (
        df[label_col]
        if not counts.empty and counts.min() >= 2
        else None
    )
    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
    label_col: str = LABEL_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/validation/test splits.

    The test split is carved first, then the validation split is taken from
    the remaining data.  If a class has fewer than two rows, the corresponding
    split falls back to a non-stratified one.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train, val, test)`` with reset integer indexes.
    """
    if (
        not 0 <= val_size < 1
        or not 0 <= test_size < 1
        or val_size + test_size >= 1
    ):
        raise ValueError(
            "val_size and test_size must be in [0, 1) and sum to less than 1"
        )

    train_val, test = _split_once(df, test_size, seed, label_col)
    if val_size == 0:
        train, val = train_val, train_val.iloc[0:0].copy()
    else:
        val_fraction = val_size / (1.0 - test_size)
        train, val = _split_once(train_val, val_fraction, seed, label_col)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, val, test


def tokenize_lab4(text: str) -> List[str]:
    """Tokenize a tweet while preserving negation words.

    URLs, mentions and retweet markers are removed.  Tokens are lowercased,
    apostrophes are stripped from their edges, and negation contractions are
    normalized to ``not`` before stopwords are filtered.  Short tokens or
    stopwords are dropped except for common negations needed by the
    ``not_cyberbullying`` class.
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _RT_RE.sub(" ", text)

    tokens = [
        _normalize_negation(token.strip("'"))
        for token in _TOKEN_RE.findall(text)
    ]
    return [
        token
        for token in tokens
        if len(token) >= 2 and (token not in stop_words or token in NEGATIONS)
    ]


def build_vocab(
    token_lists: Iterable[Sequence[str]],
    min_freq: int = 2,
    max_size: Optional[int] = None,
) -> Dict[str, int]:
    """Build a token-to-id vocabulary.

    ``<pad>`` always gets id 0 and ``<unk>`` gets id 1.  Remaining tokens are
    ordered by decreasing frequency and then alphabetically.  ``max_size``
    includes the two special tokens.
    """
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
    candidates = [
        (token, freq)
        for token, freq in counter.items()
        if freq >= min_freq and token not in vocab
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))

    if max_size is None:
        selected = candidates
    else:
        remaining = max(int(max_size) - len(vocab), 0)
        selected = candidates[:remaining]

    for token, _ in selected:
        vocab[token] = len(vocab)
    return vocab


def encode(
    text: str,
    vocab: Dict[str, int],
    max_len: Optional[int] = 64,
) -> List[int]:
    """Convert ``text`` to token ids, truncating to ``max_len``.

    Missing and out-of-vocabulary tokens map to ``<unk>``.  The returned
    sequence is not padded.
    """
    if max_len is not None and max_len < 1:
        raise ValueError("max_len must be positive or None")

    unk_id = vocab.get("<unk>", 1)
    ids = [vocab.get(token, unk_id) for token in tokenize_lab4(text)]
    if max_len is not None:
        ids = ids[:max_len]
    return ids or [unk_id]


class TextDataset(Dataset):
    """Dataset yielding ``(input_ids, label_id)`` tensor pairs."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[Any],
        vocab: Dict[str, int],
        label_to_id: Dict[Any, int],
        max_len: Optional[int] = 64,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.label_to_id = label_to_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = encode(self.texts[index], self.vocab, self.max_len)
        label_id = self.label_to_id[self.labels[index]]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long),
        )


def collate_batch(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences with zeros and stack their labels."""
    if not batch:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
        )

    sequences = [
        torch.as_tensor(item[0], dtype=torch.long) for item in batch
    ]
    labels = torch.stack(
        [torch.as_tensor(item[1], dtype=torch.long) for item in batch]
    )

    max_len = max((len(sequence) for sequence in sequences), default=0)
    padded = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        if len(sequence):
            padded[row, : len(sequence)] = sequence

    return padded, labels


def _is_known_token(w2v: Any, token: str, key_to_index: Any) -> bool:
    """Return whether ``token`` exists in a keyed vectors object."""
    if key_to_index is not None:
        return token in key_to_index
    try:
        return token in w2v
    except TypeError:
        return False


def build_embedding_matrix(
    w2v: Any,
    vocab: Dict[str, int],
    dim: int,
) -> np.ndarray:
    """Create an embedding matrix aligned with ``vocab``.

    Row 0 (``<pad>``) is zero.  Row 1 (``<unk>``) is the mean of all known
    token vectors, or zeros when no vocabulary token is present in ``w2v``.
    Out-of-vocabulary embeddings remain zero.
    """
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    if not vocab:
        return matrix

    key_to_index = getattr(w2v, "key_to_index", None)
    known_vectors: List[np.ndarray] = []

    for token, index in vocab.items():
        if index in (0, 1) or token in ("<pad>", "<unk>"):
            continue
        if not _is_known_token(w2v, token, key_to_index):
            continue
        try:
            vector = np.asarray(w2v[token], dtype=np.float32)
        except (KeyError, IndexError):
            continue
        if vector.shape != (dim,):
            raise ValueError(
                f"Embedding for token {token!r} has shape {vector.shape}, "
                f"expected ({dim},)"
            )
        matrix[index] = vector
        known_vectors.append(vector)

    if known_vectors and matrix.shape[0] > 1:
        matrix[1] = np.mean(np.vstack(known_vectors), axis=0)

    return matrix


def mean_pool_vectors(texts: Iterable[str], w2v: Any) -> np.ndarray:
    """Average pretrained word vectors for every text.

    Texts with no known tokens are represented by a zero vector.
    """
    texts_list = list(texts)
    dim = getattr(w2v, "vector_size", None)
    if dim is None:
        dim = int(w2v.vectors.shape[1])

    matrix = np.zeros((len(texts_list), dim), dtype=np.float32)
    key_to_index = getattr(w2v, "key_to_index", None)

    for row, text in enumerate(texts_list):
        vectors = []
        for token in tokenize_lab4(text):
            if not _is_known_token(w2v, token, key_to_index):
                continue
            try:
                vectors.append(np.asarray(w2v[token], dtype=np.float32))
            except (KeyError, IndexError):
                continue
        if vectors:
            matrix[row] = np.mean(np.vstack(vectors), axis=0)

    return matrix


class BiRNNClassifier(nn.Module):
    """Bidirectional LSTM/GRU classifier with masked mean pooling."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 6,
        embedding_matrix: Optional[Any] = None,
        freeze: bool = False,
        dropout: float = 0.3,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if embedding_matrix is not None:
            matrix = torch.as_tensor(
                embedding_matrix, dtype=torch.float32
            ).detach().cpu()
            expected_shape = (vocab_size, embed_dim)
            if tuple(matrix.shape) != expected_shape:
                raise ValueError(
                    f"embedding_matrix shape {tuple(matrix.shape)} does not "
                    f"match the expected shape {expected_shape}"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(matrix)

        if freeze:
            self.embedding.weight.requires_grad = False

        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "lstm":
            rnn_class = nn.LSTM
        elif self.rnn_type == "gru":
            rnn_class = nn.GRU
        else:
            raise ValueError(
                f"Unsupported rnn_type {rnn_type!r}; use 'lstm' or 'gru'"
            )
        self.rnn = rnn_class(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of padded token ids."""
        lengths = input_ids.ne(0).sum(dim=1).clamp(min=1)
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        mask = input_ids.ne(0).unsqueeze(-1).float()
        summed = (outputs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.fc(self.dropout(pooled))


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> Dict[str, float]:
    """Compute macro precision, recall, F1 and accuracy."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    if y_true_array.size == 0 or y_pred_array.size == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

    return {
        "precision": float(
            precision_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
    }


def get_device() -> torch.device:
    """Return the best available torch device (MPS, CUDA or CPU)."""
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_rnn(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate an RNN classifier on ``loader``.

    Returns a dictionary with loss, macro precision/recall/F1, accuracy and
    the collected ``y_true``/``y_pred`` lists.
    """
    device = get_device() if device is None else torch.device(device)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    loss = total_loss / total_samples if total_samples else 0.0
    result: Dict[str, Any] = classification_metrics(y_true, y_pred)
    result["loss"] = float(loss)
    result["y_true"] = y_true
    result["y_pred"] = y_pred
    return result


def train_rnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: Optional[int] = 2,
    device: Optional[Any] = None,
    class_weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Train an RNN classifier with early stopping on validation macro-F1.

    Returns a dictionary with per-epoch ``history``, the best epoch number and
    the elapsed training time.  The best validation state is restored before
    returning.
    """
    device = get_device() if device is None else torch.device(device)
    model.to(device)

    weights = None
    if class_weights is not None:
        weights = torch.as_tensor(
            class_weights, dtype=torch.float32, device=device
        )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history: List[Dict[str, float]] = []
    best_f1 = float("-inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        train_true: List[int] = []
        train_pred: List[int] = []

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            train_true.extend(labels.detach().cpu().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

        train_loss = total_loss / total_samples if total_samples else 0.0
        train_f1 = (
            float(
                f1_score(
                    train_true,
                    train_pred,
                    average="macro",
                    zero_division=0,
                )
            )
            if total_samples
            else 0.0
        )

        val_metrics = evaluate_rnn(model, val_loader, device=device)
        val_loss = float(val_metrics["loss"])
        val_f1 = float(val_metrics["f1"])

        history.append(
            {
                "train_loss": float(train_loss),
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_f1": val_f1,
            }
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                patience is not None
                and epochs_without_improvement >= patience
            ):
                break

    train_time = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "train_time": float(train_time),
    }


def bert_available() -> bool:
    """Return whether the optional ``transformers`` package is installed."""
    try:
        return importlib.util.find_spec("transformers") is not None
    except (ImportError, ValueError):
        return False


def _import_transformers() -> Tuple[Any, Any]:
    """Lazily import BERT tokenizer/model classes with a clear error."""
    if not bert_available():
        raise RuntimeError(_TRANSFORMERS_ERROR)
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(_TRANSFORMERS_ERROR) from exc
    return AutoTokenizer, AutoModelForSequenceClassification


class BertDataset(Dataset):
    """Dataset wrapping tokenizer encodings and integer class labels."""

    def __init__(
        self,
        encodings: Dict[str, Any],
        labels: Sequence[int],
    ) -> None:
        self.encodings = {
            key: torch.as_tensor(value, dtype=torch.long)
            for key, value in encodings.items()
        }
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {
            key: value[index] for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


def _encode_bert(
    df: pd.DataFrame,
    tokenizer: Any,
    label_to_id: Dict[Any, int],
    max_len: int,
) -> BertDataset:
    """Tokenize a dataframe for a BERT-style model."""
    texts = df[TEXT_COL].astype(str).tolist()
    labels = [label_to_id[label] for label in df[LABEL_COL].tolist()]
    if texts:
        encodings = dict(
            tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )
        )
    else:
        encodings = {}
    return BertDataset(encodings, labels)


def _evaluate_bert(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate BERT once, returning loss, macro-F1 and predictions."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) for key, value in batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss
            batch_size = batch["labels"].size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            y_true.extend(batch["labels"].detach().cpu().tolist())
            y_pred.extend(
                outputs.logits.argmax(dim=1).detach().cpu().tolist()
            )

    loss = total_loss / total_samples if total_samples else 0.0
    f1 = (
        float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        if total_samples
        else 0.0
    )
    return float(loss), f1, y_true, y_pred


def train_bert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_len: int = 128,
    patience: Optional[int] = 2,
    device: Optional[Any] = None,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Fine-tune a Hugging Face sequence classifier without ``Trainer``.

    The function imports ``transformers`` lazily.  Early stopping is based on
    validation macro-F1, and the best model state is restored before test
    evaluation.

    Returns
    -------
    dict
        ``history``, ``best_epoch``, ``train_time``, ``test_metrics``,
        ``model`` and ``tokenizer``.
    """
    AutoTokenizer, AutoModelForSequenceClassification = _import_transformers()

    set_seed(seed)
    device = get_device() if device is None else torch.device(device)
    label_to_id = {label: index for index, label in enumerate(LABELS)}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = _encode_bert(train_df, tokenizer, label_to_id, max_len)
    val_dataset = _encode_bert(val_df, tokenizer, label_to_id, max_len)
    test_dataset = _encode_bert(test_df, tokenizer, label_to_id, max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS)
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history: List[Dict[str, float]] = []
    best_f1 = float("-inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        train_true: List[int] = []
        train_pred: List[int] = []

        for batch in train_loader:
            batch = {
                key: value.to(device) for key, value in batch.items()
            }
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            batch_size = batch["labels"].size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            train_true.extend(batch["labels"].detach().cpu().tolist())
            train_pred.extend(
                outputs.logits.argmax(dim=1).detach().cpu().tolist()
            )

        train_loss = total_loss / total_samples if total_samples else 0.0
        train_f1 = (
            float(
                f1_score(
                    train_true,
                    train_pred,
                    average="macro",
                    zero_division=0,
                )
            )
            if total_samples
            else 0.0
        )

        val_loss, val_f1, _, _ = _evaluate_bert(
            model, val_loader, device
        )
        history.append(
            {
                "train_loss": float(train_loss),
                "train_f1": train_f1,
                "val_loss": float(val_loss),
                "val_f1": float(val_f1),
            }
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                patience is not None
                and epochs_without_improvement >= patience
            ):
                break

    train_time = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    _, _, test_true, test_pred = _evaluate_bert(
        model, test_loader, device
    )
    test_metrics = classification_metrics(test_true, test_pred)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "train_time": float(train_time),
        "test_metrics": test_metrics,
        "model": model,
        "tokenizer": tokenizer,
    }
