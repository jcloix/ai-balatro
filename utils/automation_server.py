import socket
import json
import time
from threading import Thread
import torch
import argparse
import os
import signal

from utils.screenshots import capture_cards
from utils.inference_utils import (
    load_model,
    preprocess_image,
    infer,
    map_indices_to_labels,
)
from utils.ahk_client import AHKClient

from utils.strategies.base_strategy import StrategyRegistry
import utils.strategies.logging_card_strategy  # example strategy

from config.config import CARDS_NAMES


NAME_MODEL = "data/models/best/identity.pth"
MODIFIER_MODEL = "data/models/best/modifier.pth"


class AutomationServer:
    """
    Automation server that listens for JSON commands and automates reroll/buy actions
    using model predictions, strategies, and AHK automation.
    """

    def __init__(self, name_model_path, modifier_model_path, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        self.running = True
        self.logging_mode = False
        self.strategy_mode = False
        self.rerolls = 0
        self.context = {}

        # Initialize AHK client
        self.ahk = AHKClient()

        # Load models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[INFO] Loading models...")
        self.model_name, self.class_names_name = load_model(name_model_path, device)
        self.model_modifier, self.class_names_modifier = load_model(modifier_model_path, device)
        self.device = device
        print("[INFO] Models loaded successfully.")

        # Load reference data
        self.reference_data = self._load_reference_data(CARDS_NAMES)

        # Load strategies
        self.strategies = StrategyRegistry.get_all()
        self.current_strategy = None

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

    # -----------------------------
    # Reference data
    # -----------------------------
    def _load_reference_data(self, path):
        if not os.path.exists(path):
            print(f"[WARN] Reference data not found: {path}")
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
                print(f"[INFO] Loaded reference data with {len(data)} entries.")
                return data
        except Exception as e:
            print(f"[ERROR] Failed to load reference data: {e}")
            return {}

    # -----------------------------
    # Signal handler
    # -----------------------------
    def _signal_handler(self, sig, frame):
        print("[INFO] Ctrl+C pressed. Shutting down server...")
        self.running = False

    # -----------------------------
    # Server setup
    # -----------------------------
    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(1.0)  # allows periodic checks for self.running
        print(f"[✓] Server listening on {self.host}:{self.port}")

        # Start automation loop in background
        Thread(target=self.automation_loop, daemon=True).start()

        # Command listener loop
        try:
            while self.running:
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue

                print(f"[INFO] Connection from {addr}")
                data = conn.recv(4096).decode()

                if not data:
                    conn.close()
                    continue

                try:
                    command = json.loads(data)
                except json.JSONDecodeError:
                    print("[ERROR] Invalid JSON command received.")
                    self._send_json(conn, {"status": "error", "message": "Invalid JSON"})
                    conn.close()
                    continue

                print(f"[CMD] Received command: {command}")
                response = self.handle_command(command)
                self._send_json(conn, response)
                conn.close()
        finally:
            server.close()
            print("[INFO] Server shut down cleanly.")

    def _send_json(self, conn, data):
        try:
            conn.send(json.dumps(data).encode())
        except Exception as e:
            print(f"[WARN] Failed to send response: {e}")

    # -----------------------------
    # Command handling
    # -----------------------------
    def handle_command(self, command):
        cmd = command.get("command")

        if cmd == "get_strategies":
            strategies = {name: s.description for name, s in self.strategies.items()}
            return {"status": "ok", "strategies": strategies}

        elif cmd == "apply_strategy":
            strategy_name = command.get("strategy_name")
            rerolls = int(command.get("rerolls", 1))
            money = command.get("money", 0)
            current_rerolls = command.get("current_rerolls", 0)

            if strategy_name not in self.strategies:
                return {"status": "error", "message": f"Unknown strategy: {strategy_name}"}

            self.strategy_mode = True
            self.logging_mode = False
            self.rerolls = rerolls
            self.context = {"money": money, "current_rerolls": current_rerolls}
            self.current_strategy = self.strategies[strategy_name]

            print(f"[MODE] Strategy '{strategy_name}' activated for {rerolls} rerolls.")
            return {"status": "ok", "strategy": strategy_name, "rerolls": rerolls}

        elif cmd == "start_logging":
            self.strategy_mode = False
            self.logging_mode = True
            self.rerolls = int(command.get("rerolls", 1))
            print(f"[MODE] Logging mode activated for {self.rerolls} rerolls.")
            return {"status": "ok", "mode": "logging"}

        elif cmd == "stop":
            self.strategy_mode = False
            self.logging_mode = False
            print("[MODE] All modes stopped.")
            return {"status": "ok", "message": "Stopped"}

        elif cmd == "pause":
            self.running = False
            print("[STATE] Automation paused.")
            return {"status": "ok", "message": "Paused"}

        elif cmd == "resume":
            self.running = True
            print("[STATE] Automation resumed.")
            return {"status": "ok", "message": "Resumed"}

        else:
            print(f"[WARN] Unknown command: {cmd}")
            return {"status": "error", "message": f"Unknown command: {cmd}"}

    # -----------------------------
    # Automation loop
    # -----------------------------
    def automation_loop(self):
        while self.running:
            if self.logging_mode or self.strategy_mode:
                screenshots = capture_cards()
                
                cards = self.predict_cards(screenshots)

                # Logging mode
                if self.logging_mode:
                    for card in cards:
                        print(f"[LOG] {card}")
                    self.rerolls -= 1
                    if self.rerolls <= 0:
                        self.logging_mode = False
                        print("[INFO] Logging complete.")
                    else:
                        self.ahk.reroll()

                # Strategy mode
                if self.strategy_mode and self.current_strategy:
                    print(f"[STRATEGY] Executing '{self.current_strategy.name}'...")
                    self.current_strategy.handle_cards(cards, self.context, self.ahk)

                    self.rerolls -= 1
                    if self.rerolls <= 0:
                        self.strategy_mode = False
                        print("[INFO] Strategy complete.")
                    else:
                        self.ahk.reroll()

                time.sleep(2)
            else:
                time.sleep(0.5)

    # -----------------------------
    # Predict cards from screenshots
    # -----------------------------
    def predict_cards(self, screenshots):
        results = []
        for i,img in enumerate(screenshots):
            #filename = f"image-{i}.png"
            #img.save(os.path.join("data", filename))
            img_tensor = preprocess_image(img).to(self.device)

            # Predict with both models
            res_name = infer(self.model_name, img_tensor, topk=1)
            res_mod = infer(self.model_modifier, img_tensor, topk=1)

            mapped_name = map_indices_to_labels(res_name, self.class_names_name)
            mapped_mod = map_indices_to_labels(res_mod, self.class_names_modifier)

            name_label = (
                mapped_name.get("identification", [{"label": "Unknown"}])[0]["label"]
                if mapped_name else "Unknown"
            )
            modifier_label = (
                mapped_mod.get("modifier", [{"label": "None"}])[0]["label"]
                if mapped_mod else "None"
            )

            card_info = self.reference_data.get(name_label, {})
            card_type = card_info.get("type", "Unknown")
            rarity = card_info.get("rarity", "Unknown")

            results.append({
                "name": name_label,
                "modifier": modifier_label,
                "type": card_type,
                "rarity": rarity
            })

        return results


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automation server for Balatro AI.")
    parser.add_argument("--name-model", type=str, default=NAME_MODEL, help="Name model path")
    parser.add_argument("--modifier-model", type=str, default=MODIFIER_MODEL, help="Modifier model path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")

    args = parser.parse_args()

    server = AutomationServer(
        name_model_path=args.name_model,
        modifier_model_path=args.modifier_model,
        host=args.host,
        port=args.port,
    )
    server.start_server()
