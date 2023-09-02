import os
# from pydub import AudioSegment
import subprocess

idx=0
for file in os.listdir('/data/hypertext/sharpwang/dataset/TTS/yujie'):
    # print(file.split('.'))
    path_to_ffmpeg = "/data/hypertext/sharpwang/TTS/ffmpeg-6.0-amd64-static/ffmpeg"  # 替换为真实路径
    input_file = "/data/hypertext/sharpwang/dataset/TTS/yujie/" + file  # 替换为真实路径
    output_file = "/data/hypertext/sharpwang/dataset/TTS/yujie_wav_1/"+ file.split('.')[0] + '.wav' # 替换为真实路径

    command = [path_to_ffmpeg, "-i", input_file, output_file]
    subprocess.run(command, check=True)
    # idx+=1
    # exit()

print(len(os.listdir('/data/hypertext/sharpwang/dataset/TTS/yujie')))
print(len(os.listdir('/data/hypertext/sharpwang/dataset/TTS/yujie_wav_1')))