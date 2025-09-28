# tests/test_data_loader.py
from unittest.mock import patch, mock_open
from annotate_dataset.data_loader import (
    load_labels,
    list_images,
    parse_ids,
    build_maps,
    get_unlabeled_groups,
    compute_unique_by_type
)

# ---------------------------
# load_labels
# ---------------------------
@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"a": 1}')
def test_load_labels_exists(mock_file, mock_exists):
    labels = load_labels()
    assert labels == {"a": 1}

@patch("os.path.exists", return_value=False)
def test_load_labels_not_exists(mock_exists):
    labels = load_labels()
    assert labels == {}

# ---------------------------
# list_images
# ---------------------------
@patch("os.listdir", return_value=["a.png", "b.txt", "c.JPG", "d.jpeg"])
def test_list_images(mock_listdir):
    images = list_images()
    assert sorted(images) == ["a.png", "c.JPG", "d.jpeg"]

# ---------------------------
# parse_ids
# ---------------------------
def test_parse_ids():
    cluster, group = parse_ids("cluster10_card5_sample.png")
    assert cluster == 10
    assert group == 5

# ---------------------------
# build_maps
# ---------------------------
def test_build_maps():
    images = ["cluster1_card1_sample.png", "cluster1_card2_sample.png", "cluster2_card3_sample.png"]
    cg_map, cl_map = build_maps(images)

    # card_group_map keys = card group IDs
    assert set(cg_map.keys()) == {1, 2, 3}

    # cluster_map keys = cluster IDs
    assert set(cl_map.keys()) == {1, 2}

    # card_group_map contents
    assert cg_map[1] == ["cluster1_card1_sample.png"]
    assert cg_map[2] == ["cluster1_card2_sample.png"]
    assert cg_map[3] == ["cluster2_card3_sample.png"]

    # cluster_map contents
    assert set(cl_map[1]) == {"cluster1_card1_sample.png", "cluster1_card2_sample.png"}
    assert cl_map[2] == ["cluster2_card3_sample.png"]

# ---------------------------
# get_unlabeled_groups
# ---------------------------
def test_get_unlabeled_groups():
    card_group_map = {1: ["a.png", "b.png"], 2: ["c.png"]}
    labels = {"a.png": {}}
    unlabeled = get_unlabeled_groups(card_group_map, labels)
    assert unlabeled == [1, 2]

# ---------------------------
# compute_unique_by_type
# ---------------------------
def test_compute_unique_by_type():
    labels = {
        "a.png": {"name": "CardA", "type": "Joker"},
        "b.png": {"name": "CardB", "type": "Planet"},
        "c.png": {"name": "CardC", "type": "Joker"}
    }
    result = compute_unique_by_type(labels)
    assert result["Joker"] == {"CardA", "CardC"}
    assert result["Planet"] == {"CardB"}
    # Ensure other types exist and are empty
    for t in ["Tarot", "Spectral"]:
        assert result[t] == set()
