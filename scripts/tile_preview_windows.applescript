on tilePreviewProcess(processID, windowPosition, windowSize)
    tell application "System Events"
        repeat 100 times
            set matchingProcesses to every application process whose unix id is processID
            if (count of matchingProcesses) is 1 then
                set targetProcess to item 1 of matchingProcesses
                tell targetProcess
                    if (count of windows) is greater than 0 then
                        set position of window 1 to windowPosition
                        set size of window 1 to windowSize
                        return
                    end if
                end tell
            end if
            delay 0.1
        end repeat
    end tell
    error "Timed out waiting for Preview process " & processID
end tilePreviewProcess

on run arguments
    if (count of arguments) is not 2 then
        error "Expected accepted-PDF and production-PDF Preview process IDs"
    end if

    set acceptedProcessID to (item 1 of arguments) as integer
    set productionProcessID to (item 2 of arguments) as integer

    tell application "Finder"
        set screenBounds to bounds of window of desktop
    end tell

    set screenLeft to item 1 of screenBounds
    set screenTop to item 2 of screenBounds
    set screenRight to item 3 of screenBounds
    set screenBottom to item 4 of screenBounds
    set topInset to 28
    set bottomInset to 80
    set gapWidth to 8
    set usableWidth to screenRight - screenLeft
    set paneWidth to (usableWidth - gapWidth) div 2
    set usableHeight to screenBottom - screenTop - topInset - bottomInset

    set leftPosition to {screenLeft, screenTop + topInset}
    set rightPosition to {screenLeft + paneWidth + gapWidth, screenTop + topInset}
    set paneSize to {paneWidth, usableHeight}

    my tilePreviewProcess(acceptedProcessID, leftPosition, paneSize)
    my tilePreviewProcess(productionProcessID, rightPosition, paneSize)

    tell application "System Events"
        set frontmost of first application process whose unix id is productionProcessID to true
        set frontmost of first application process whose unix id is acceptedProcessID to true
    end tell
end run
