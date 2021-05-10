@echo off
echo Creating list of image names
(for %%i in (*.png) do @echo file '%%i') > mylist.txt
echo Creating movie
ffmpeg -r 71 -f concat -i mylist.txt ../videos/estimatelaserboundary-circlepopcount.mp4