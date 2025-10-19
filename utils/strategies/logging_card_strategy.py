from utils.strategies.base_strategy import BaseStrategy, StrategyRegistry

@StrategyRegistry.register
class LoggingCardStrategy(BaseStrategy):
    name = "Logging Card Buyer"
    description = "Simply log the cards"

    def handle_cards(self, cards, context, ahk_client):
        for i, card in enumerate(cards):
            print(f"[STRATEGY] card {i}: {card['name']} (modifier: {card['modifier']})")
