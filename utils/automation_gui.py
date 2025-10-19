import tkinter as tk
from tkinter import ttk, messagebox
import socket
import json

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000


# -----------------------------
# Networking helpers
# -----------------------------
def send_command(command, expect_response=False):
    try:
        s = socket.socket()
        s.connect((SERVER_HOST, SERVER_PORT))
        s.send(json.dumps(command).encode())

        response = None
        if expect_response:
            data = s.recv(4096)
            if data:
                try:
                    response = json.loads(data.decode())
                except json.JSONDecodeError:
                    response = {"status": "error", "message": "Invalid response format"}

        s.close()
        return response
    except Exception as e:
        print(f"[ERROR] Could not send command: {e}")
        return {"status": "error", "message": str(e)}


# -----------------------------
# Server communication
# -----------------------------
def fetch_strategies():
    """Ask the server for all registered strategies."""
    resp = send_command({"command": "get_strategies"}, expect_response=True)
    if resp and resp.get("status") == "ok":
        return resp.get("strategies", {})
    else:
        print(f"[WARN] Could not fetch strategies: {resp}")
        return {}


# -----------------------------
# Command actions
# -----------------------------
def start_strategy():
    strategy_name = strategy_var.get()
    money = money_var.get()
    current_rerolls = current_rerolls_var.get()
    rerolls_wanted = rerolls_wanted_var.get()

    if not strategy_name or strategy_name.startswith("("):
        messagebox.showwarning("Missing strategy", "Please select a strategy first.")
        return

    command = {
        "command": "apply_strategy",
        "strategy_name": strategy_name,
        "money": money,
        "current_rerolls": current_rerolls,
        "rerolls": rerolls_wanted,
    }
    send_command(command)


def stop_automation():
    send_command({"command": "stop"})


def pause_automation():
    send_command({"command": "pause"})


def resume_automation():
    send_command({"command": "resume"})


# -----------------------------
# GUI setup
# -----------------------------
root = tk.Tk()
root.title("Automation Controller")

# Context Frame (top section)
context_frame = ttk.LabelFrame(root, text="Context")
context_frame.pack(padx=10, pady=5, fill="x")

ttk.Label(context_frame, text="Money:").grid(row=0, column=0, padx=5, pady=5)
money_var = tk.StringVar(value="0")
ttk.Entry(context_frame, textvariable=money_var, width=10).grid(row=0, column=1, padx=5, pady=5)

ttk.Label(context_frame, text="Current Rerolls:").grid(row=0, column=2, padx=5, pady=5)
current_rerolls_var = tk.StringVar(value="0")
ttk.Entry(context_frame, textvariable=current_rerolls_var, width=10).grid(row=0, column=3, padx=5, pady=5)

ttk.Label(context_frame, text="Rerolls Wanted:").grid(row=0, column=4, padx=5, pady=5)
rerolls_wanted_var = tk.StringVar(value="5")
ttk.Entry(context_frame, textvariable=rerolls_wanted_var, width=10).grid(row=0, column=5, padx=5, pady=5)


# Strategy Frame
strategy_frame = ttk.LabelFrame(root, text="Strategy")
strategy_frame.pack(padx=10, pady=5, fill="x")

ttk.Label(strategy_frame, text="Strategy:").pack(side="left", padx=5)
strategy_var = tk.StringVar()
strategy_combo = ttk.Combobox(strategy_frame, textvariable=strategy_var, state="readonly", width=25)
strategy_combo.pack(side="left", padx=5)

ttk.Button(strategy_frame, text="Apply Strategy", command=start_strategy).pack(side="left", padx=10)


# Control Frame
control_frame = ttk.LabelFrame(root, text="Control")
control_frame.pack(padx=10, pady=5, fill="x")

ttk.Button(control_frame, text="Stop", command=stop_automation).pack(side="left", padx=5)
ttk.Button(control_frame, text="Pause", command=pause_automation).pack(side="left", padx=5)
ttk.Button(control_frame, text="Resume", command=resume_automation).pack(side="left", padx=5)


# -----------------------------
# Initialize dropdown
# -----------------------------
def update_strategies():
    strategies = fetch_strategies()
    names = list(strategies.keys())
    if names:
        strategy_combo["values"] = names
        strategy_combo.current(0)
    else:
        strategy_combo["values"] = ["(no strategies available)"]
        strategy_combo.current(0)


update_strategies()

root.mainloop()
