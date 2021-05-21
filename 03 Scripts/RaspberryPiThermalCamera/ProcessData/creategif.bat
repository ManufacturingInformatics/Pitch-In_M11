@echo off
for /f %%i in ('cd') do set FD=%%~nxi
(for %%i in (*.png) do @echo file '%%i') > mylist.txt
ffmpeg -r 60 -f concat -i mylist.txt pi-camera-frames-%FD%.gif
