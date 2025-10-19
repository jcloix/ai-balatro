; click_position.ahk

if !A_Args[1] || !A_Args[2]
{
    MsgBox "Usage: click_position.ahk <x> <y>"
    ExitApp
}
CoordMode("Mouse", "Screen")
x := A_Args[1]
y := A_Args[2]

Click(x, y)
ExitApp
