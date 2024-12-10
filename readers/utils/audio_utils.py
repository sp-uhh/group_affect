import subprocess

def addAudioToVideo(audioSource, videoSource, savePath):
    cmd = f"ffmpeg -i {videoSource} -i {audioSource} -c copy -map 0:v -map 0:a -map 1:a {savePath}"
    subprocess.run(cmd)