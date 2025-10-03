import pytest
from train_model.factory import HEAD_REGISTRY, create_head

# Import the head classes to trigger the @register_head decorator
from train_model.heads import IdentificationHead, ModifierHead

@pytest.fixture(autouse=True)
def reset_registry():
    # Clear and re-populate the registry without instantiating heads
    HEAD_REGISTRY.clear()
    HEAD_REGISTRY["identification"] = IdentificationHead
    HEAD_REGISTRY["modifier"] = ModifierHead
    yield
    HEAD_REGISTRY.clear()  # optional cleanup

def test_registry_contains_real_heads():
    assert "identification" in HEAD_REGISTRY
    assert "modifier" in HEAD_REGISTRY
    assert HEAD_REGISTRY["identification"] is IdentificationHead
    assert HEAD_REGISTRY["modifier"] is ModifierHead

def test_create_head_real_classes_minimal_config():
    config = {"batch_size": 2, "val_split": 0.1}
    id_head = create_head("identification", config)
    mod_head = create_head("modifier", config)
    assert isinstance(id_head, IdentificationHead)
    assert isinstance(mod_head, ModifierHead)
    assert id_head.batch_size == 2
    assert mod_head.batch_size == 2
    assert id_head.val_split == 0.1
    assert mod_head.val_split == 0.1

def test_create_head_unknown_raises():
    with pytest.raises(ValueError):
        create_head("nonexistent_head")
