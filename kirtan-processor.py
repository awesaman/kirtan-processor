#!/usr/bin/env python
from pydub import AudioSegment, silence, effects
import time
from os import mkdir, listdir, walk, cpu_count, path
# import banidb # for naming shabads later
import tkinter
from tkinter import filedialog as fd
root = tkinter.Tk()
import concurrent.futures as conc
from itertools import repeat
# from multiprocessing import Process

# global chosen_dir_path

def format_time(t):
    mins = t//60000
    if mins > 59:
        return f'{mins//60}:{mins%60:02}:{(t//1000)%60:02}'
    return f'{mins}:{(t//1000)%60:02}'
def format_ms(start,end):
    return f'{format_time(start)} - {format_time(end)}'

def slash(dir_path):
    # windows vs. mac stuff
    if dir_path.find('/') != -1:
        return '/'
    return '\\'

def rename_output(file_path, chosen_dir_path):
    sl = slash(chosen_dir_path)

    path_after_chosen_dir = file_path[len(chosen_dir_path)+1:]
    path_after_chosen_dir = path_after_chosen_dir.replace(slash(chosen_dir_path), ' - ')
    
    # remove duplication of ZOOMxxxx in name
    # print(path_after_chosen_dir[-11:-7])
    if len(path_after_chosen_dir) >= 22 and path_after_chosen_dir[-22:-15] == ' - ZOOM':
        path_after_chosen_dir = path_after_chosen_dir[:-11]
    return chosen_dir_path + sl + 'edited' + sl + path_after_chosen_dir

def edit_tracks(tup, chosen_dir_path):
    dir_path, subdirs, filenames = tup

    # windows vs. mac stuff
    # if dir_path.find('\\') != -1 and dir_path[-1] != '\\':
    #     dir_path += '\\'
    # elif dir_path.find('/') != -1 and dir_path[-1] != '/':
    #     dir_path += '/'
    if dir_path[-1] == '/' or dir_path[-1] == '\\':
        dir_path = dir_path[:-1]
    if dir_path[-6:] == 'edited': return
    dir_path += slash(dir_path)

    
    tracks = {dir_path + tr[:8] for tr in filenames if tr[:4] == 'ZOOM'}
    formats = {tr[-4:] for tr in filenames if tr[:4] == 'ZOOM'}
    # formats = {tr[-4:] for tr in filenames if tr[:4] == 'ZOOM' and tr[8:15] != 'segment'}
    # for tr in filenames:
    #     if tr[8:15] == 'segment':
    #         tracks.discard(dir_path + tr[:8])
    formats.discard('hprj')
    if len(tracks) == 0:
        print(f'WARNING: no tracks found in {dir_path}')
        return
    if len(formats) != 1:
        print(f'ERROR: please make all audio files the same format in {dir_path}')
        return
    track_format = list(formats)[0]
    for track in tracks:
        if path.exists(f'{rename_output(track, chosen_dir_path)} - Segment 1.mp3'):
            print(f'{rename_output(track, chosen_dir_path)} has already been edited, it will be skipped.')
            tracks.remove(track)
    print(f'NOTE: found {len(tracks)} unedited tracks in {dir_path}')

    for track in tracks:
        # rename_output(track,chosen_dir_path)
        # print("created: %s" % time.ctime(path.getmtime(f'{track}LR.WAV')))
        # this didn't work because metadata was destroyed during download
        # next step is to use fake vaari data and match up vaari time with timestamp

        unsegmented_track = f'{track}unsegmented{track_format}'
        if unsegmented_track[len(dir_path):] in filenames:
            audio = AudioSegment.from_file(unsegmented_track)

        else:
            # merge possible separate audio tracks
            print(f'importing {track[len(dir_path):]}...')
            sangat_track = f'{track}_LR{track_format}'
            if sangat_track[len(dir_path):] not in filenames:
                sangat_tracks = [tr for tr in filenames if tr.startswith(sangat_track[len(dir_path):-4] + '-')]
                print(f'concatenating {sangat_tracks}')
                sangat_audios = [AudioSegment.from_file(dir_path + tr) for tr in sorted(sangat_tracks)]
                sangat = sum(sangat_audios, AudioSegment.empty())
            else:
                sangat = AudioSegment.from_file(sangat_track)
            
            tabla_track = f'{track}_Tr1{track_format}'
            if tabla_track[len(dir_path):] not in filenames:
                tabla_tracks = [tr for tr in filenames if tr.startswith(tabla_track[len(dir_path):-4] + '-')]
                print(f'concatenating {tabla_tracks}')
                tabla_audios = [AudioSegment.from_file(dir_path + tr) for tr in sorted(tabla_tracks)]
                tabla = sum(tabla_audios, AudioSegment.empty())
            else:
                tabla = AudioSegment.from_file(tabla_track)
            
            kirtani_track = f'{track}_Tr3{track_format}'
            if kirtani_track[len(dir_path):] not in filenames:
                kirtani_tracks = [tr for tr in filenames if tr.startswith(kirtani_track[len(dir_path):-4] + '-')]
                print(f'concatenating {kirtani_tracks}')
                kirtani_audios = [AudioSegment.from_file(dir_path + tr) for tr in sorted(kirtani_tracks)]
                kirtani = sum(kirtani_audios, AudioSegment.empty())
            else:
                kirtani = AudioSegment.from_file(kirtani_track)
            
            # feel free to manually adjust these constants
            print(f'normalizing {track[len(dir_path):]}...')
            target_db = -3
            sangat = effects.normalize(sangat) + target_db - 1
            tabla = effects.normalize(tabla) + target_db - 2
            kirtani = effects.normalize(kirtani) + target_db

            print(f'mixing {track[len(dir_path):]}...')
            audio = kirtani.overlay(tabla).overlay(sangat)
            # include this line if you want an uncut version
            # audio.export(f'{track}unsegmented{track_format}',format=track_format[1:].lower())

        print(f'segmenting {track[len(dir_path):]}...')
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
            # this print statement shows doesn't show the final cuts, only the prelims
            # print(format_ms(start,end))
        
        # rare case where kirtani is doing alaap at the end and it's too soft to pick up
        ls,le = final_cuts[-1]
        # print(len(final_cuts), le - ls, 10*60000)
        if len(final_cuts) >= 2 and le - ls < 10*60000:
            final_cuts = final_cuts[:-1]
            final_cuts[-1][1] = le

        print(f'{track[len(dir_path):]} segments to be exported: {[format_ms(start,end) for start,end in final_cuts]}')
        prev_length = 0
        
        for i, (start, end) in enumerate(final_cuts):
            length = end - start
            if length < 17*60000:
                print(f'Possible Error: {format_ms(start,end)} length is only {format_time(length)}.')
            # only give this error when we can tell if it is a Simran or not (based on vaari data)
            # if length > 37*60000:
            #     print(f'Possible Error: {format_ms(start,end)} length is quite long at {format_time(length)}')
            elif length + 120000 < prev_length:
                print(f'Possible Error: {format_ms(start,end)} length is shorter than the previous track by {format_time(prev_length-length)}.')
            # audio[max(start-1000,0):end].export(f'{track}segment_{i}.mp3',format='mp3')
            audio[start:end].export(f'{rename_output(track, chosen_dir_path)} - Segment {i+1}.mp3',format='mp3')
            prev_length = length

        print(f'created {len(final_cuts)} {track[len(dir_path):]} segments...\n')

if __name__ == '__main__':
    startTime = time.time()
    # with conc.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    with conc.ProcessPoolExecutor() as executor:
        # chosen_dir_path = '/Users/aman/Downloads/temp/Fri'
        chosen_dir_path = fd.askdirectory(parent=root, initialdir='.', title='Please select a directory')

        if chosen_dir_path[-1] == '/' or chosen_dir_path[-1] == '\\':
            chosen_dir_path = chosen_dir_path[:-1]
        if ('edited' not in listdir(chosen_dir_path)):
            mkdir(f'{chosen_dir_path}{slash(chosen_dir_path)}edited')

        print(f'\nyou chose {chosen_dir_path}')
        all_subdirs = tuple(walk(chosen_dir_path))
        results = executor.map(edit_tracks, all_subdirs, repeat(chosen_dir_path))
        
        # this is just to view errors
        # for res in results:
        #     print(res)
        # executor.submit(edit_tracks, all_subdirs[2])
        # print(all_subdirs)
        # edit_tracks(all_subdirs[2])
    # processes = []

    # for sd in all_subdirs:
    #     p = Process(target=edit_tracks,args=[sd, chosen_dir_path])
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
            
    # note that an increase of x decibels is 10^(x/10) fold increase
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(round(executionTime)))