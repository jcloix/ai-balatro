#Requires AutoHotkey v2
#SingleInstance Force
CoordMode("Mouse", "Screen")

; -----------------------------
; Configuration
; -----------------------------
REROLL_X := 700
REROLL_Y := 590
PYTHON_HOST := "127.0.0.1"
PYTHON_PORT := 50007

; -----------------------------
; Function to send message to Python via TCP
; -----------------------------
sendToPython(msg) {
    ; Escape single quotes for PowerShell
    msgEsc := StrReplace(msg, "'", "''")

    ; Build the PowerShell command string
    psBody := "$client=New-Object System.Net.Sockets.TcpClient('127.0.0.1',50007); "
    psBody .= "$writer=New-Object System.IO.StreamWriter($client.GetStream()); "
    psBody .= "$writer.AutoFlush=$true; "
    psBody .= "$writer.WriteLine('" msgEsc "'); "
    psBody .= "$writer.Close(); $client.Close()"

    ; Wrap psBody in quotes properly using Chr(34)
    psCmd := "powershell -NoProfile -Command " Chr(34) psBody Chr(34)

    ; Run hidden
    Run(psCmd, , "Hide")
}


global Running := false
global RemainingClicks := 0

; -----------------------------
; Main clicker hotkey: Ctrl + Shift + ;
; -----------------------------
^+::
{
    global Running, RemainingClicks

    if (Running) {
        ; Stop the clicking
        Running := false
        RemainingClicks := 0
        return
    }

    ; Start clicking
    result := InputBox("How many times do you want to click?", "Clicker")
    if result.Result != "OK"
        return

    clicks := result.Value
    if !RegExMatch(clicks, "^\d+$") {
        MsgBox "Please enter a positive number."
        return
    }

    RemainingClicks := Integer(clicks)
    Running := true

    ; Start timer for clicking every 2000ms
    SetTimer(ClickStep, 3000)
}

; -----------------------------
; Timer callback for clicks
; -----------------------------
ClickStep() {
    global Running, RemainingClicks

    if !Running || RemainingClicks <= 0 {
        SetTimer(ClickStep, 0)  ; stop timer
        Running := false
        ToolTip("")
        return
    }
    sendToPython("capture")
	Sleep 1000  ; wait for cards to refresh

    Click
    RemainingClicks--
}