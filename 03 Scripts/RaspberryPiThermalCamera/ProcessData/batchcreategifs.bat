@echo off
for /f %%f in ('dir /ad /b') do (
	for /f %%A in ('dir /a-d-s-h /b %%f ^| find /v /c ""') do set cnt=%%A
	(for %%i in (%ADDR%/*.png) do @echo file '%%f/%%i') > mylist.txt
	ffmpeg -r %cnt%/120 -f concat -i mylist.txt pi-camera-frames-%%f.gif
)