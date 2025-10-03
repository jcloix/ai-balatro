
HEAD_REGISTRY = {}

def register_head(name):
    def decorator(cls):
        HEAD_REGISTRY[name] = cls
        return cls
    return decorator


def create_head(name: str, config=None):
    if name not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type: {name}. Available: {list(HEAD_REGISTRY.keys())}")
    return HEAD_REGISTRY[name](name, config)
