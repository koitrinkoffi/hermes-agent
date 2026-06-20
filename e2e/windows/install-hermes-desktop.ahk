#Requires AutoHotkey v2.0
#SingleInstance Force


logPath := A_Args.Length >= 1 ? A_Args[1] : "ahk.log"

Log(text) {
    ToolTip(text)
    FileAppend(Format("{}`n", text), '*')
    FileAppend(Format("{}`n", text), logPath)
}

OnError(LogError)

LogError(err, mode) {
    FileAppend(
        "Unhandled error: " err.Message "`n",
        "*"
    )
    FileAppend(
        "Unhandled error: " err.Message "`n",
        logPath
    )
    return -1  ; suppress the standard error dialog
}

SetWorkingDir(A_ScriptDir)
CoordMode("Pixel", "Screen")
CoordMode("Mouse", "Screen")


ClickWithMarker(x, y, button := "Left") {
    Click(x, y, button)

    Sleep(10)
    MouseMove(30, 30)
    Log(Format("Clicking at {1}, {2}", x, y))
    size := 20
    g := Gui("-Caption +AlwaysOnTop +ToolWindow")
    g.BackColor := "Red"
    g.Show(Format(
        "x{} y{} w{} h{} NoActivate"
        , x - size // 2
        , y - size // 2
        , size
        , size
    ))
    hRegion := DllCall(
        "CreateEllipticRgn"
        , "Int", 0
        , "Int", 0
        , "Int", size
        , "Int", size
        , "Ptr"
    )
    DllCall("SetWindowRgn", "Ptr", g.Hwnd, "Ptr", hRegion, "Int", true)
    WinSetTransparent(255, g.Hwnd)
    SetTimer(() => g.Destroy(), -500)
}


ClickCenterOfImageInWindow(winTitle, imageFile, timeoutMs := 10000, intervalMs := 250)
{
    WinGetPos(&wx, &wy, &ww, &wh, winTitle)

    hBitmap := LoadPicture(imageFile)

    bm := Buffer(32, 0) ; BITMAP structure on x64
    DllCall("GetObject", "Ptr", hBitmap, "Int", bm.Size, "Ptr", bm)

    width := NumGet(bm, 4, "Int")
    height := NumGet(bm, 8, "Int")


    startTime := A_TickCount

    timeLeft := timeoutMs - (A_TickCount - startTime)
    while (timeLeft > 0)
    {
        try
        {
            if ImageSearch(&x, &y, wx, wy, wx + ww, wy + wh, Format("*10 {}", imageFile))
            {
                ClickWithMarker(x + Floor(width / 2), y + Floor(height / 2))
                Log("Found button!")
                return
            }
        }

        Sleep intervalMs
        timeLeft := timeoutMs - (A_TickCount - startTime)
        ToolTip(Format("Searching for button {} in window {}...  {}s left", imageFile, winTitle, Round(timeLeft / 1000, 2)))
    }

    throw Error(Format("Failed to find button {} in window {}", imageFile, winTitle))
}


Log("Waiting for the installer window to appear...")
winTitle := "Hermes"
try {
    WinWait(winTitle, , 30)
} catch {
    throw Error("Hermes installer window did not appear within 30s")
}
WinGetPos(&x, &y, &w, &h, winTitle)
Log(Format("Window found at x={1} y={2} w={3} h={4}`n", x, y, w, h))

ClickCenterOfImageInWindow(winTitle, "install-button.png")

ClickCenterOfImageInWindow(winTitle, "install-button.png", 1000 * 60 * 8)


; done
ExitApp(0)