from pydub import AudioSegment, silence, effects
import time
from os import listdir #, path
# import banidb # for naming shabads later
import tkinter
from tkinter import filedialog as fd
root = tkinter.Tk()

def format_time(t):
    mins = t//60000
    if mins > 59:
        return f'{mins//60}:{mins%60:02}:{(t//1000)%60:02}'
    return f'{mins}:{(t//1000)%60:02}'
def format_ms(start,end):
    return f'{format_time(start)} - {format_time(end)}'

startTime = time.time()

# dir_path=input('Which folder would you like to process?')
# dir_path = '/Users/aman/Downloads/raw3'
dir_path = fd.askdirectory(parent=root,initialdir='.', title='Please select a directory')
# root.quit()
print(f'\nfound path {dir_path}')
raw_tracks = listdir(dir_path)

# windows vs. mac stuff
if dir_path.find('\\') != -1 and dir_path[-1] != '\\':
    dir_path += '\\'
elif dir_path.find('/') != -1 and dir_path[-1] != '/':
    dir_path += '/'

tracks = {dir_path + tr[:9] for tr in raw_tracks if tr[:4] == 'ZOOM'}
formats = {tr[-4:] for tr in raw_tracks if tr[:4] == 'ZOOM'}
if len(formats) != 1:
    print('ERROR: please make all audio files the same format')
    exit()
track_format = list(formats)[0]

for track in tracks:
    # print("created: %s" % time.ctime(path.getmtime(f'{track}LR.WAV')))
    # this didn't work because metadata was destroyed during download
    # next step is to use fake vaari data and match up vaari time with timestamp

    unsegmented_track = f'{track}unsegmented{track_format}'
    if unsegmented_track[len(dir_path):] in raw_tracks:
        audio = AudioSegment.from_file(unsegmented_track)

    else:
        # merge possible separate audio tracks
        sangat_track = f'{track}LR{track_format}'
        if sangat_track[len(dir_path):] not in raw_tracks:
            sangat_tracks = [tr for tr in raw_tracks if tr.startswith(sangat_track[len(dir_path):-4] + '-')]
            print(f'concatenating {sangat_tracks}')
            sangat_audios = [AudioSegment.from_file(dir_path + tr) for tr in sorted(sangat_tracks)]
            sangat = sum(sangat_audios, AudioSegment.empty())
        else:
            sangat = AudioSegment.from_file(sangat_track)
        
        tabla_track = f'{track}Tr1{track_format}'
        if tabla_track[len(dir_path):] not in raw_tracks:
            tabla_tracks = [tr for tr in raw_tracks if tr.startswith(tabla_track[len(dir_path):-4] + '-')]
            print(f'concatenating {tabla_tracks}')
            tabla_audios = [AudioSegment.from_file(dir_path + tr) for tr in sorted(tabla_tracks)]
            tabla = sum(tabla_audios, AudioSegment.empty())
        else:
            tabla = AudioSegment.from_file(tabla_track)
        
        kirtani_track = f'{track}Tr3{track_format}'
        if kirtani_track[len(dir_path):] not in raw_tracks:
            kirtani_tracks = [tr for tr in raw_tracks if tr.startswith(kirtani_track[len(dir_path):-4] + '-')]
            print(f'concatenating {kirtani_tracks}')
            kirtani_audios = [AudioSegment.from_file(dir_path + tr) for tr in sorted(kirtani_tracks)]
            kirtani = sum(kirtani_audios, AudioSegment.empty())
        else:
            kirtani = AudioSegment.from_file(kirtani_track)
        
        # feel free to manually adjust these constants
        print(f'normalizing {track[len(dir_path):-1]}...')
        target_db = -3
        sangat = effects.normalize(sangat) + target_db - 1
        tabla = effects.normalize(tabla) + target_db - 2
        kirtani = effects.normalize(kirtani) + target_db

        print(f'mixing {track[len(dir_path):-1]}...')
        audio = kirtani.overlay(tabla).overlay(sangat)
        # include this line if you want an uncut version
        # audio.export(f'{track}unsegmented{track_format}',format=track_format[1:].lower())

    print(f'segmenting {track[len(dir_path):-1]}...')
    dBFS=audio.dBFS

    # note: dBFS-20 will make nonsilent segments shorter than dBFS-23
    segments = silence.detect_nonsilent(audio, min_silence_len=4000, silence_thresh=dBFS-21, seek_step=2000)
    # segments = silence.detect_nonsilent(audio, min_silence_len=5000, silence_thresh=dBFS-22, seek_step=2000)
    final_cuts = []
    prev_end = 0
    min_time_between_vaaris = 10000
    dropout = 60000
    for start,end in segments:
        # drop out anything less than 60 seconds
        if end - start < dropout:
            continue

        # the longer the previous segment is, the less likely we are to merge the current segment into it
        if final_cuts:
            prev_length = final_cuts[-1][1]-final_cuts[-1][0]

            # case if prev segment is less than 16 minutes
            if prev_length < 15*60000 or start - prev_end < min_time_between_vaaris:
                final_cuts[-1][1] = end
            else:
                final_cuts.append([start,end])
        
        else:
            final_cuts.append([start,end])
        prev_end = end
        print(format_ms(start,end))
    
    # very rare case where kirtani is doing alaap at the end and it's too soft to pick up
    ls,le = final_cuts[-1]
    print(len(final_cuts), le - ls, 10*60000)
    if len(final_cuts) >= 2 and le - ls < 10*60000:
        final_cuts = final_cuts[:-1]
        final_cuts[-1][1] = le

    print([format_ms(start,end) for start,end in final_cuts])
    prev_length = 0
    for i,(start,end) in enumerate(final_cuts):
        length = end-start
        if length < 17*60000:
            print(f'Possible Error: {format_ms(start,end)} length is only {format_time(length)}.')
        # only give this error when we can tell if it is a Simran or not (based on vaari data)
        # if length > 37*60000:
        #     print(f'Possible Error: {format_ms(start,end)} length is quite long at {format_time(length)}')
        elif length + 120000 < prev_length:
            print(f'Possible Error: {format_ms(start,end)} length is shorter than the previous track by {format_time(prev_length-length)}.')
        # audio[max(start-1000,0):end].export(f'{track}segment_{i}.mp3',format='mp3')
        audio[start:end].export(f'{track}segment_{i}.mp3',format='mp3')
        prev_length = length

    print(f'created {len(final_cuts)} {track[len(dir_path):-1]} segments...\n')

    # note that an increase of x decibels is 10^(x/10) fold increase
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))