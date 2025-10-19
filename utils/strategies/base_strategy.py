from abc import ABC, abstractmethod

class StrategyRegistry:
    _strategies = {}

    @classmethod
    def register(cls, strategy_cls):
        """Decorator for registering strategy classes."""
        instance = strategy_cls()
        cls._strategies[instance.name] = instance
        return strategy_cls

    @classmethod
    def get_all(cls):
        return cls._strategies

    @classmethod
    def get(cls, name):
        return cls._strategies.get(name)


class BaseStrategy(ABC):
    name = "Base"
    description = "Abstract base strategy."

    @abstractmethod
    def handle_cards(self, cards, context, ahk_client):
        pass
