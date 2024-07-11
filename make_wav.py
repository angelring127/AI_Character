import pysubs2
import moviepy.editor as mp

# 자막 파일과 영상 파일 경로
subtitles_file = "subtitle.ass"
video_file = "movie.mkv"

# 자막 파일 로드
subs = pysubs2.load(subtitles_file)

speech_segments = []

for line in subs:
      start_time = line.start / 1000  # milliseconds to seconds
      end_time = line.end / 1000
      speech_segments.append((start_time, end_time, line.text))

# 구간 출력
for segment in speech_segments:
    print(f"Start: {segment[0]}s, End: {segment[1]}s, Text: {segment[2]}")

# 오디오 추출
video = mp.VideoFileClip(video_file)

for i, segment in enumerate(speech_segments):
    start_time = segment[0]
    end_time = segment[1]
    output_filename = f"./movie_06/output_audio_06_{i}.wav"
    print(output_filename)
    
    # 해당 구간의 오디오 추출
    audio_clip = video.subclip(start_time, end_time).audio
    audio_clip.write_audiofile(output_filename, codec='pcm_s16le')  # WAV 형식으로 저장
    
    print(f"Extracted audio: {output_filename}")

print("All audio segments have been extracted.")