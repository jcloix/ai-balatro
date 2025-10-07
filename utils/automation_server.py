# file: automation_server.py
import socket
import json
import time
from threading import Thread
from models.models import load_checkpoint

# Dummy function for predictions (replace with your model)
def predict_cards(screenshots):
    # Return a list of predicted cards
    return [{"name": "CardA", "rarity": "Rare"}, {"name": "CardB", "rarity": "Common"}]

# Dummy strategy: buy all Rare cards
def apply_strategy(cards):
    actions = []
    for i, card in enumerate(cards):
        if card["rarity"] == "Rare":
            actions.append({"type": "click", "position": (100 + i*50, 200)})
    return actions

class AutomationServer:
    def __init__(self, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        self.running = False
        self.logging_mode = False
        self.strategy_mode = False
        self.rerolls = 0

    def start_server(self):
        self.running = True
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(1)
        print(f"Server listening on {self.host}:{self.port}")
        Thread(target=self.automation_loop, daemon=True).start()

        while True:
            conn, addr = server.accept()
            print(f"Connection from {addr}")
            data = conn.recv(1024).decode()
            if not data:
                conn.close()
                continue
            command = json.loads(data)
            print(f"Received command: {command}")
            self.handle_command(command)
            conn.send(b"OK")
            conn.close()

    def handle_command(self, command):
        cmd = command.get("command")
        if cmd == "start_logging":
            self.logging_mode = True
            self.rerolls = command.get("rerolls", 1)
        elif cmd == "apply_strategy":
            self.strategy_mode = True
            self.rerolls = command.get("rerolls", 1)
        elif cmd == "stop":
            self.logging_mode = False
            self.strategy_mode = False
        elif cmd == "pause":
            self.running = False
        elif cmd == "resume":
            self.running = True

    def automation_loop(self):
        while True:
            if self.running and (self.logging_mode or self.strategy_mode):
                # Take screenshots (replace with your function)
                screenshots = ["screenshot1.png", "screenshot2.png"]
                
                # Predict cards
                cards = predict_cards(screenshots)
                
                # Log cards
                if self.logging_mode:
                    for card in cards:
                        print(f"[LOG] {card}")
                    self.rerolls -= 1
                    if self.rerolls <= 0:
                        self.logging_mode = False

                # Apply strategy
                if self.strategy_mode:
                    actions = apply_strategy(cards)
                    for action in actions:
                        # Send action to AHK
                        self.send_to_ahk(action)
                    self.rerolls -= 1
                    if self.rerolls <= 0:
                        self.strategy_mode = False

                time.sleep(2)  # wait 2 seconds between rerolls
            else:
                time.sleep(0.5)

    def send_to_ahk(self, action):
        ahk_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ahk_socket.connect(("127.0.0.1", 6000))
        ahk_socket.send(json.dumps(action).encode())
        ahk_socket.close()


if __name__ == "__main__":
    server = AutomationServer()
    server.start_server()
