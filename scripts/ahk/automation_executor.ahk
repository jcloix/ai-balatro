; file: automation_executor.ahk
; Listens on port 6000 for JSON commands from Python
; Requires ahk2exe with TCP support or use a wrapper like nircmd or Python's pyautogui

#Persistent
port := 6000

; Using TCP server with AHK wrapper (or external tool)
Loop {
    ; Example: receive JSON from Python (replace with actual socket listener)
    ; Simulate click command
    if FileExist("action.json") {
        FileRead, jsonData, action.json
        obj := {}
        try obj := JSON_Load(jsonData)
        x := obj.position[1]
        y := obj.position[2]
        Click, %x%, %y%
        FileDelete, action.json
    }
    Sleep, 100
}

JSON_Load(json) {
    ; Simple parser for demo (or use AHK JSON library)
    ; Replace with proper parser for production
    return StrReplace(json, "`"", "") ; dummy placeholder
}
