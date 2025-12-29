# convert images to mp4
ffmpeg -framerate 25 -i debug_imgs/%06d.jpg -b 2000k video8_1.mp4

# add audio to mp4
ffmpeg -i video_08_01.mp4 -i 0703_1_sync.mp3 -map 0:v -map 1:a -c:v copy -c:a aac output_08_01.mp4

# combine 2 mp4.
ffmpeg -i vertice_gt.mp4 -i vertice_pred.mp4 -filter_complex "[0:v][1:v]hstack" -c:a copy output.mp4