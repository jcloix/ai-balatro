# tests/test_dataset_hash_storage.py
import os
import json
import tempfile
import shutil
import pytest
import imagehash
from PIL import Image

import build_dataset.hash_storage as hs


@pytest.fixture(autouse=True)
def clean_globals_and_tmpfile(monkeypatch):
    # Reset globals
    hs.seen_hashes.clear()
    hs.cluster_counter = 0
    hs.card_group_counter = 0

    # Temporary file to isolate test writes
    tmpdir = tempfile.mkdtemp()
    tmpfile = os.path.join(tmpdir, "hashes.json")
    monkeypatch.setattr(hs, "HASH_FILE", tmpfile)

    yield

    shutil.rmtree(tmpdir)


def test_save_and_load_roundtrip():
    # Create dummy hashes
    img = Image.new("RGB", (8, 8), color="red")
    ph = imagehash.phash(img)
    ah = imagehash.average_hash(img)
    dh = imagehash.dhash(img)

    hs.seen_hashes.append({
        "ph": ph,
        "ah": ah,
        "dh": dh,
        "cluster": 1,
        "card_group": 2,
        "filename": "dummy.png"
    })

    # Save to file
    hs.save_hashes()
    assert os.path.exists(hs.HASH_FILE)

    # Clear globals, then reload
    hs.seen_hashes.clear()
    hs.cluster_counter = 0
    hs.card_group_counter = 0
    hs.load_hashes()

    assert len(hs.seen_hashes) == 1
    entry = hs.seen_hashes[0]

    assert str(entry["ph"]) == str(ph)
    assert str(entry["ah"]) == str(ah)
    assert str(entry["dh"]) == str(dh)
    assert entry["cluster"] == 1
    assert entry["card_group"] == 2
    assert entry["filename"] == "dummy.png"

    assert hs.cluster_counter == 1
    assert hs.card_group_counter == 2


def test_load_with_empty_file():
    # Should not crash or modify state if file is empty
    with open(hs.HASH_FILE, "w") as f:
        f.write("[]")

    hs.load_hashes()
    assert hs.seen_hashes == []
    assert hs.cluster_counter == 0
    assert hs.card_group_counter == 0


def test_load_with_legacy_card_group_key():
    # Old data uses "cluster" instead of "card_group"
    data = [{
        "ph": "0000ffffffff0000",
        "ah": "0000ffffffff0000",
        "dh": "0000ffffffff0000",
        "cluster": 5,
    }]
    with open(hs.HASH_FILE, "w") as f:
        json.dump(data, f)

    hs.load_hashes()
    assert len(hs.seen_hashes) == 1
    entry = hs.seen_hashes[0]

    # Card group should fall back to cluster
    assert entry["cluster"] == 5
    assert entry["card_group"] == 5
