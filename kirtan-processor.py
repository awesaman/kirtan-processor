from pydub import AudioSegment, silence, effects
import time
from os import listdir #, path
# import tkinter
# from tkinter import filedialog as fd
# root = tkinter.Tk()
# import banidb # for naming shabads later

startTime = time.time()
# shabads = banidb.search("sGrb")

dir_path=input('Which folder would you like to process?')
# /Users/aman/Downloads/raw2
# dir_path = fd.askdirectory(parent=root,initialdir='.', title='Please select a directory')
# root.destroy
print(f'found path {dir_path}')
raw_tracks = listdir(dir_path)
if dir_path.find('\\') != -1 and dir_path[-1] != '\\':
    dir_path += '\\'
elif dir_path.find('/') != -1 and dir_path[-1] != '/':
    dir_path += '/'
tracks = {dir_path + tr[:9] for tr in raw_tracks if tr[:4] == 'ZOOM'} # exclude hidden files
for track in tracks:
    # print("created: %s" % time.ctime(path.getmtime(f'{track}LR.WAV')))
    # this didn't work because metadata was destroyed during download
    # next step is to use fake vaari data and match up vaari time with timestamp

    # merge possible separate audio tracks
    sangat_track = f'{track}LR.WAV'
    if sangat_track[len(dir_path):] not in raw_tracks:
        sangat_tracks = [tr for tr in raw_tracks if tr.startswith(sangat_track[len(dir_path):-4])]
        sangat_audios = [AudioSegment.from_file(dir_path + tr) for tr in sorted(sangat_tracks)]
        sangat = sum(sangat_audios, AudioSegment.empty())
    else:
        sangat = AudioSegment.from_file(sangat_track)
    
    tabla_track = f'{track}Tr1.WAV'
    if tabla_track[len(dir_path):] not in raw_tracks:
        tabla_tracks = [tr for tr in raw_tracks if tr.startswith(tabla_track[len(dir_path):-4])]
        tabla_audios = [AudioSegment.from_file(dir_path + tr) for tr in sorted(tabla_tracks)]
        tabla = sum(tabla_audios, AudioSegment.empty())
    else:
        tabla = AudioSegment.from_file(tabla_track)
    
    kirtani_track = f'{track}Tr3.WAV'
    if kirtani_track[len(dir_path):] not in raw_tracks:
        kirtani_tracks = [tr for tr in raw_tracks if tr.startswith(kirtani_track[len(dir_path):-4])]
        kirtani_audios = [AudioSegment.from_file(dir_path + tr) for tr in sorted(kirtani_tracks)]
        kirtani = sum(kirtani_audios, AudioSegment.empty())
    else:
        kirtani = AudioSegment.from_file(kirtani_track)
    
    # feel free to manually adjust these constants
    print(time.time()-startTime)
    print('normalizing...')
    target_db = -3
    sangat = effects.normalize(sangat) + target_db - 1
    tabla = effects.normalize(tabla) + target_db - 2
    kirtani = effects.normalize(kirtani) + target_db

    print('mixing...')
    audio = kirtani.overlay(tabla).overlay(sangat)
    print(f'working on {track[len(dir_path):-1]} segments...')
    
    # include this line if you want an uncut version
    # audio.export(f'{track}normalized_uncut.WAV',format='wav')
    # replace sangat, kirtan, tabla with this next line if we have exported the uncut version
    # audio = AudioSegment.from_file(f'{track}normalized_uncut.WAV')

    dBFS=audio.dBFS

    # what will actually be used
    segments = silence.detect_nonsilent(audio, min_silence_len=10000, silence_thresh=dBFS-22, seek_step=3000)
    final_cuts = []
    # gap = 18000000000
    prev_end = 0
    min_time_between_vaaris = 20000
    for start,end in segments:
        # drop out anything less than 20 seconds
        if end - start < min_time_between_vaaris:
            continue

        # the longer the previous segment is, the less likely we are to merge the current segment into it
        if prev_end:
            prev_length = final_cuts[-1][1]-final_cuts[-1][0]
            
            # case if prev segment is less than 15 minutes or gap is longer than 20 seconds
            if prev_length < 15*60000 or start - prev_end < min_time_between_vaaris:
                # print('not appending')
                # print(prev_length, 15*60000)
                # print(min_time_between_vaaris, start - prev_end)
                final_cuts[-1][1] = end
            else:
                final_cuts.append([start,end])
        
        # case if first segment of recording
        else:
            final_cuts.append([start,end])
        prev_end = end
        print('{}:{} - {}:{}'.format(start//60000,(start//1000)%60,end//60000,(end//1000)%60))
    
    # final_cuts may have less entries because of merging
    print(final_cuts)
    prev_length = 0
    for i,(start,end) in enumerate(final_cuts):
        length = end-start
        if length < 17*60000:
            print('Possible Error: length is only {} minutes.'.format(length/60000))
        # only give this error when we can tell if it is a Simran or not (based on vaari data)
        # if length > 37*60000:
        #     print('Possible Error: length is {} minutes, quite long'.format(length/60000))
        if length + 120000 < prev_length:
            print('Possible Error: length is shorter than the previous track by {} minutes.'.format((prev_length-length)/60000))
        audio[start:end].export('{}mixed_segment_{}.mp3'.format(track,i),format='mp3')
        prev_length = length

    print(f'created {track[len(dir_path):-1]} segments...\n')

    # note that an increase of x decibels is 10^(x/10) fold increase
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))