import pytest

# Import the registry functions
from train_model.factory import HEAD_REGISTRY, register_head, create_head  # adjust import

# -----------------------------
# Dummy classes for testing
# -----------------------------
class DummyHead:
    def __init__(self, name, config=None):
        self.name = name
        self.config = config

class AnotherHead:
    def __init__(self, name, config=None):
        self.name = name
        self.config = config

# -----------------------------
# Tests
# -----------------------------
def test_register_head_and_create_head():
    # Clear registry before test
    HEAD_REGISTRY.clear()

    # Register DummyHead
    @register_head("dummy")
    class Dummy(DummyHead):
        pass

    # Check registry
    assert "dummy" in HEAD_REGISTRY
    assert HEAD_REGISTRY["dummy"] is Dummy

    # Create instance
    instance = create_head("dummy", config={"param": 42})
    assert isinstance(instance, Dummy)
    assert instance.name == "dummy"
    assert instance.config == {"param": 42}

def test_multiple_heads_registration():
    # Clear registry
    HEAD_REGISTRY.clear()

    @register_head("dummy")
    class Dummy(DummyHead): pass

    @register_head("another")
    class Another(AnotherHead): pass

    # Check registry
    assert "dummy" in HEAD_REGISTRY
    assert "another" in HEAD_REGISTRY

    # Create both
    dummy_instance = create_head("dummy")
    another_instance = create_head("another")

    assert isinstance(dummy_instance, Dummy)
    assert isinstance(another_instance, Another)

def test_create_head_unknown_name_raises():
    HEAD_REGISTRY.clear()
    with pytest.raises(ValueError) as e:
        create_head("nonexistent")
    assert "Unknown head type" in str(e.value)
