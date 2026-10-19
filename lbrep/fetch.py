"""Fetch single weight tensors from Hugging Face safetensors files via HTTP range requests.

No full-model download: read the index, the 8-byte header length, the JSON header, then the
tensor's byte range. Every model is pinned to a commit sha (models.json).
"""
import hashlib, json, struct
from pathlib import Path

import numpy as np
import requests

CACHE = Path(__file__).resolve().parent.parent / 'cache'
HF = 'https://huggingface.co'
S = requests.Session()
S.headers['User-Agent'] = 'littlebit-replication/1.0'


def _url(repo, sha, fname):
    return f'{HF}/{repo}/resolve/{sha}/{fname}'


def _get(url, start=None, end=None):
    h = {} if start is None else {'Range': f'bytes={start}-{end}'}
    for attempt in range(5):
        try:
            r = S.get(url, headers=h, timeout=120, allow_redirects=True)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 4:
                raise


def index(repo, sha):
    """Map tensor name -> shard filename (single-file models included)."""
    p = CACHE / repo.replace('/', '__') / sha / 'index.json'
    if p.exists():
        return json.loads(p.read_text())
    try:
        wm = json.loads(_get(_url(repo, sha, 'model.safetensors.index.json')))['weight_map']
    except requests.HTTPError:
        hdr = header(repo, sha, 'model.safetensors')
        wm = {k: 'model.safetensors' for k in hdr if k != '__metadata__'}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(wm))
    return wm


_HDR = {}


def header(repo, sha, shard):
    key = (repo, sha, shard)
    if key not in _HDR:
        url = _url(repo, sha, shard)
        n = struct.unpack('<Q', _get(url, 0, 7))[0]
        h = json.loads(_get(url, 8, 8 + n - 1))
        h['__offset__'] = 8 + n
        _HDR[key] = h
    return _HDR[key]


DT = {'BF16': 'bf16', 'F16': np.float16, 'F32': np.float32}


def tensor(repo, sha, name):
    """Return float32 numpy array and sha256 of its raw bytes."""
    shard = index(repo, sha)[name]
    h = header(repo, sha, shard)
    meta = h[name]
    a, b = meta['data_offsets']
    off = h['__offset__']
    raw = _get(_url(repo, sha, shard), off + a, off + b - 1)
    assert len(raw) == b - a, f'short read {name}'
    digest = hashlib.sha256(raw).hexdigest()
    if meta['dtype'] == 'BF16':
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        arr = u.view(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=DT[meta['dtype']]).astype(np.float32)
    return arr.reshape(meta['shape']), digest
