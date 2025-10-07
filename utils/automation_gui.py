# file: automation_gui.py
import tkinter as tk
from tkinter import ttk
import socket
import json

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000

def send_command(command):
    try:
        s = socket.socket()
        s.connect((SERVER_HOST, SERVER_PORT))
        s.send(json.dumps(command).encode())
        s.close()
    except Exception as e:
        print(f"Error sending command: {e}")

def start_logging():
    rerolls = int(logging_rerolls.get())
    send_command({"command": "start_logging", "rerolls": rerolls})

def start_strategy():
    rerolls = int(strategy_rerolls.get())
    send_command({"command": "apply_strategy", "rerolls": rerolls})

def stop_automation():
    send_command({"command": "stop"})

def pause_automation():
    send_command({"command": "pause"})

def resume_automation():
    send_command({"command": "resume"})

# Tkinter GUI
root = tk.Tk()
root.title("Automation Controller")

# Logging Frame
logging_frame = ttk.LabelFrame(root, text="Logging")
logging_frame.pack(padx=10, pady=5, fill="x")
ttk.Label(logging_frame, text="Rerolls:").pack(side="left")
logging_rerolls = tk.StringVar(value="5")
ttk.Entry(logging_frame, textvariable=logging_rerolls, width=5).pack(side="left", padx=5)
ttk.Button(logging_frame, text="Start Logging", command=start_logging).pack(side="left", padx=5)

# Strategy Frame
strategy_frame = ttk.LabelFrame(root, text="Strategy")
strategy_frame.pack(padx=10, pady=5, fill="x")
ttk.Label(strategy_frame, text="Rerolls:").pack(side="left")
strategy_rerolls = tk.StringVar(value="5")
ttk.Entry(strategy_frame, textvariable=strategy_rerolls, width=5).pack(side="left", padx=5)
ttk.Button(strategy_frame, text="Apply Strategy", command=start_strategy).pack(side="left", padx=5)

# Control Frame
control_frame = ttk.LabelFrame(root, text="Control")
control_frame.pack(padx=10, pady=5, fill="x")
ttk.Button(control_frame, text="Stop", command=stop_automation).pack(side="left", padx=5)
ttk.Button(control_frame, text="Pause", command=pause_automation).pack(side="left", padx=5)
ttk.Button(control_frame, text="Resume", command=resume_automation).pack(side="left", padx=5)

root.mainloop()
