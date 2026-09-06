"""Corpus to tokens. Own tokenizer, trained from scratch on the corpus itself.

    python -m pretrain.data tokenizer     train the 16k byte level BPE
    python -m pretrain.data shard         tokenize every parquet into uint16 shards

Input   corpus/raw/*.parquet   (FineWeb-Edu sample files, text column)
Output  pretrain/tokenizer.json
        corpus/tokens/train_*.npy, corpus/tokens/val.npy   (np.uint16)
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "corpus" / "raw"
TOK = ROOT / "corpus" / "tokens"
TOKENIZER = ROOT / "pretrain" / "tokenizer.json"
VOCAB = 16384
EOT = "<|endoftext|>"
SHARD_TOKENS = 100_000_000


def parquet_files():
    return sorted(RAW.glob("*.parquet"))


def iter_texts(path, batch_size=2048):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        for t in batch.column("text").to_pylist():
            if t:
                yield t


# ---------------------------------------------------------------- tokenizer
def train_tokenizer(n_docs=400_000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    def sample():
        n = 0
        for f in parquet_files():
            for t in iter_texts(f):
                yield t
                n += 1
                if n >= n_docs:
                    return

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    t0 = time.time()
    tok.train_from_iterator(sample(), trainer=trainer)
    tok.save(str(TOKENIZER))
    print(f"tokenizer: {tok.get_vocab_size()} tokens from {n_docs} docs in {time.time()-t0:.0f}s -> {TOKENIZER}")
    probe = "The Prime Minister of Greece said on Tuesday that the budget would be ready by December."
    ids = tok.encode(probe).ids
    print(f"probe: {len(probe.split())} words -> {len(ids)} tokens ({len(ids)/len(probe.split()):.2f} per word)")


# ---------------------------------------------------------------- sharding
_tok = None
_eot = None


def _init():
    global _tok, _eot
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(str(TOKENIZER))
    _eot = _tok.token_to_id(EOT)


def _encode_batch(texts):
    out = []
    for enc in _tok.encode_batch(texts):
        out.extend(enc.ids)
        out.append(_eot)
    return np.asarray(out, dtype=np.uint16)


def shard(workers=None):
    TOK.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    files = parquet_files()
    print(f"sharding {len(files)} parquet files with {workers} workers")
    buf, n_buf, shard_id, total, t0 = [], 0, 0, 0, time.time()

    def flush(name):
        nonlocal buf, n_buf
        arr = np.concatenate(buf)
        np.save(TOK / f"{name}.npy", arr)
        print(f"  {name}.npy  {len(arr)/1e6:.1f}M tokens  ({total/1e6:.0f}M total, {total/(time.time()-t0)/1e6:.2f}M tok/s)")
        buf, n_buf = [], 0

    def batches():
        for f in files:
            batch = []
            for t in iter_texts(f):
                batch.append(t)
                if len(batch) >= 256:
                    yield batch
                    batch = []
            if batch:
                yield batch

    with Pool(workers, initializer=_init) as pool:
        for arr in pool.imap(_encode_batch, batches(), chunksize=4):
            buf.append(arr)
            n_buf += len(arr)
            total += len(arr)
            if n_buf >= SHARD_TOKENS:
                flush(f"train_{shard_id:05d}")
                shard_id += 1
    if buf:
        flush("val")            # the tail becomes validation, never trained on
    (TOK / "meta.json").write_text(json.dumps({"tokens": total, "shards": shard_id,
                                               "vocab": VOCAB}), encoding="utf-8")
    print(f"done: {total/1e9:.2f}B tokens in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "tokenizer":
        train_tokenizer()
    elif cmd == "shard":
        shard()
    else:
        print(__doc__)
