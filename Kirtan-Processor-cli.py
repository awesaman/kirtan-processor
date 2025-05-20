#!/usr/bin/env python
from pydub import AudioSegment, silence, effects
import time
from os import mkdir, listdir, walk, path, cpu_count
from tqdm import tqdm

# import banidb # for naming shabads later
import tkinter
from tkinter import filedialog as fd

root = tkinter.Tk()
import concurrent.futures as conc
from itertools import repeat, chain
from gc import collect
import threading

# --- ADJUST AS NEEDED ---
DB_MAP = {
    "kirtani": -3,
    "tabla": -4,
    "sangat": -4,
}
# ------------------------
MIC_MAP = {
    "Tr3": "kirtani",  # for ZOOM H6 Handy Recorder
    "Tr1": "tabla",  # for ZOOM H6 Handy Recorder
    "LR": "sangat",  # for ZOOM H6 Handy Recorder
    "3": "kirtani",
    "5": "tabla",
    "1-2": "sangat",
}

ONE_MIN = 60000
mics = []
INPUT_FORMAT = ".WAV"


def format_time(t) -> str:
    """Formats num milliseconds to mm:ss or hh:mm:ss"""
    seconds = int(t / 1000)
    minutes = seconds // 60
    seconds = seconds % 60
    
    if minutes > 59:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def format_ms(start, end):
    return f"{format_time(start)} - {format_time(end)}"


def slash(dir_path) -> str:
    """Returns \\ or / based on mac or windows"""
    # if dir_path.find('/') != -1:
    # return '/'
    # return '\\'
    return "/"


def export_audio_slice(audio, s, e, name):
    """
    Slices an AudioSegment from 3 seconds before x to 3 seconds after y,
    respecting the AudioSegment's boundaries.
    """

    start = max(0, s - 3000)  # Ensure start is not negative
    end = min(len(audio), e + 3000)  # Ensure end does not exceed audio length
    audio[start:end].export(name, format="mp3")


def get_track_name(track_file_path, dir_path) -> str:
    """Returns prefix (usually ZOOM) + 4 digit track number given the full file path by cutting off the length of the directory"""
    return track_file_path[len(dir_path) :]


def get_export_name(track_file_path, chosen_dir_path, prefix) -> str:
    """Returns what the final export should be named"""
    sl = slash(chosen_dir_path)

    path_after_chosen_dir = track_file_path[len(chosen_dir_path) + 1 :]
    path_after_chosen_dir = path_after_chosen_dir.replace(sl, " - ")

    # remove duplication of prefix in final file name
    # e.g. Sat - Evening - ZOOM0028 - ZOOM0028 converted to Sat - Evening - ZOOM0028
    if path_after_chosen_dir.count(prefix) > 1:
        end = path_after_chosen_dir.rfind(prefix)
        path_after_chosen_dir = path_after_chosen_dir[: end - 3]
    return chosen_dir_path + sl + "edited" + sl + path_after_chosen_dir


def get_mic_names(filenames):
    mics = set()
    for file in filenames:
        start = file.rfind("_")
        end = file.rfind(".")
        # files can be formatted like so if they exceed 2GB: ZOOM0011_LR-0001.WAV
        if file[end - 5 : end - 1] == "-000":
            end -= 5
        mics.add(file[start + 1 : end])
    return {mic: 0 for mic in mics}


def import_audio(dir_path, track_name):
    try:
        print(f"    fetching {get_track_name(track_name, dir_path)}")
        return AudioSegment.from_file(track_name)
    except Exception as e:
        print(f"ERROR: could not import {track_name}: {e}")
        return None


def get_track_audio(track_file_path, mic, input_format, dir_path, filenames):
    track_name = f"{track_file_path}_{mic}{input_format}"
    files_lowercase = map(lambda f: f.lower(), filenames)
    if get_track_name(track_name, dir_path).lower() in files_lowercase:
        return mic, import_audio(dir_path, track_name)

    # likely left recorder running for a long time
    # need to concatenate ZOOMxxxx_mic-0001, ZOOMxxxx_mic-0002, etc,
    tracks = [
        tr for tr in filenames if tr.startswith(track_name[len(dir_path) : -4] + "-")
    ]  # -4 cuts out the '.wav'

    tracks.sort()
    print(f"[1.5/5] concatenating {tracks}")
    audios = [import_audio(dir_path, dir_path + tr) for tr in tracks]
    audios = sum(audios, AudioSegment.empty())
    return mic, audios


def discard_other_files(filenames):
    return [f for f in filenames if "0" in f and f[-4:].lower() == INPUT_FORMAT.lower()]


def edit_tracks(tup, chosen_dir_path, mics):
    dir_path, subdirs, filenames = tup
    dir_path = dir_path.replace("\\", "/")
    # unless there are >=1000 files, there will always be a 0
    filenames = discard_other_files(filenames)
    if not filenames:
        print(f"WARNING: no valid files found in {dir_path}")
        return

    # ensuring dir_path is in expected format
    if dir_path[-1] != "/":
        dir_path += slash(dir_path)
    # this is our output path, we don't want to process it
    if dir_path[-7:] == "edited/":
        return

    # track will be numbered based on last 4 digits behind the last underscore
    # accounts for cases like
    #   ZOOM0008_Tr1 -> prefix = ZOOM, id = 0008
    #   Custom_0008_Tr1 -> prefix = Custom_, id = 0008
    #   042724_203144_Tr1 -> prefix = 042724_20, id = 3144         this isn't preferred, but should do the job
    prefix_end_index = filenames[0].rfind("_") - 4
    prefix = filenames[0][:prefix_end_index]

    tracks = {
        dir_path + tr[: prefix_end_index + 4]
        for tr in filenames
        if tr.startswith(prefix)
    }
    if len(tracks) == 0:
        print(f"WARNING: no tracks found in {dir_path}")
        return

    edited_tracks = set()
    for track in tracks:
        if path.exists(
            f"{get_export_name(track, chosen_dir_path, prefix)} - Segment 1.mp3"
        ):
            print(
                f"INFO: {get_export_name(track, chosen_dir_path, prefix)} has already been edited, it will be skipped."
            )
            edited_tracks.add(track)
    tracks -= edited_tracks
    print(f"INFO: found {len(tracks)} unedited tracks in {dir_path}")

    for track in tracks:
        # NOTE: these comments are attempts to automatically name the tracks
        # get_export_name(track,chosen_dir_path, prefix)
        # print("created: %s" % time.ctime(path.getmtime(f'{track}LR.WAV'))
        # this didn't work because metadata was destroyed during download
        # next step is to use fake vaari data and match up vaari time with timestamp

        unsegmented_track = f"{track}unsegmented{INPUT_FORMAT}"
        if get_track_name(unsegmented_track, dir_path) in filenames:
            audio = AudioSegment.from_file(unsegmented_track)

        else:
            # merge possible separate audio tracks
            print(f"[1/5] importing {get_track_name(track, dir_path)}")
            # use thread-level concurrency within 1 core for I/O intensive tasks
            audios = {}
            futures = []
            with conc.ThreadPoolExecutor(max_workers=len(mics)) as executor:
                for mic in mics.keys():
                    future = executor.submit(
                        get_track_audio, track, mic, INPUT_FORMAT, dir_path, filenames
                    )
                    futures.append(future)

                for future in tqdm(conc.as_completed(futures), total=len(futures)):
                    try:
                        mic, audio = future.result()  # Get the result of the future
                        audios[mic] = audio
                    except Exception as e:
                        # Log the exception for this particular mic
                        print(f"Error processing audio for mic {mic}: {e}")

            # feel free to manually adjust these constants
            print(f"[2/5] normalizing {get_track_name(track, dir_path)}")
            for mic in mics:
                if mics[mic] != "n":
                    # user specified not to normalize and adjust this mic
                    audios[mic] = effects.normalize(audios[mic]) + mics[mic]
                    if mic == "LR" or mic == "1-2":  # always sangat mics
                        audios[mic] = effects.low_pass_filter(
                            audios[mic], 6000
                        )  # attempt at dulling chainne

            print(f"[3/5] mixing {get_track_name(track, dir_path)}")
            audios = list(audios.values())
            audio = audios[0]
            for segment in audios[1:]:
                audio = audio.overlay(segment)
            # garbage collection to avoid using too much RAM
            del segment
            audios.clear()
            collect()

            # for convenience during testing (CHECK_BEFORE_SUBMIT):
            # include this line if you want an uncut version
            # audio.export(f'{track}unsegmented{imported_audio_format}',format=INPUT_FORMAT[1:])

        print(f"[4/5] segmenting {get_track_name(track, dir_path)}")

        # NOTE: dBFS-20 will make nonsilent segments shorter than dBFS-23
        dBFS = audio.dBFS
        auto_detected_segments = silence.detect_nonsilent(
            audio, min_silence_len=4000, silence_thresh=dBFS - 21, seek_step=2000
        )
        # considered other options such as:
        # auto_detected_segments = silence.detect_nonsilent(audio, min_silence_len=5000, silence_thresh=dBFS-22, seek_step=2000)

        # --- SETTINGS for calculating final_segments
        min_time_between_vaaris = 10000  # in ms
        min_vaari_length = 15 * ONE_MIN  # in ms
        dropout = ONE_MIN

        final_segments = []
        prev_end = 0
        for start, end in auto_detected_segments:
            # drop out anything less than 60 seconds
            if end - start < dropout:
                continue

            # the longer the previous segment is, the less likely we are to merge the current segment into it
            if final_segments:
                prev_length = final_segments[-1][1] - final_segments[-1][0]

                if (
                    prev_length < min_vaari_length
                    or start - prev_end < min_time_between_vaaris
                ):
                    # extend prev segment
                    final_segments[-1][1] = end
                else:
                    # create new segment
                    final_segments.append([start, end])

            else:
                final_segments.append([start, end])
            prev_end = end

        # handles rare case where kirtani is doing alaap at the end and it's too soft to pick up
        if len(final_segments) == 0:
            print(
                f"WARNING (please check): no segments found in {get_track_name(track, dir_path)}"
            )
        else:
            ls, le = final_segments[-1]
            if len(final_segments) >= 2 and le - ls < 10 * ONE_MIN:
                final_segments = final_segments[:-1]
                final_segments[-1][1] = le
            print(
                f"    {get_track_name(track, dir_path)} segments to be exported: {[format_ms(start,end) for start,end in final_segments]}"
            )
        print(f"[5/5] exporting {get_track_name(track, dir_path)}")

        prev_length = 0
        for i, (start, end) in enumerate(final_segments):
            length = end - start
            if length < 17 * ONE_MIN:
                print(
                    f"WARNING (please check): {format_ms(start,end)} length is only {format_time(length)}."
                )
            # only give this error when we can tell if it is a Simran or not (based on vaari data)
            # if length > 37 * ONE_MIN:
            #     print(f'WARNING (please check): {format_ms(start,end)} length is quite long at {format_time(length)}')
            elif length + 2 * ONE_MIN < prev_length:
                # warning since every vaari is generally longer than the one before it
                print(
                    f"WARNING (please check): {format_ms(start,end)} length is shorter than the previous track by {format_time(prev_length-length)}."
                )
            export_audio_slice(
                audio,
                start,
                end,
                f"{get_export_name(track, chosen_dir_path, prefix)} - Segment {i+1}.mp3",
            )
            prev_length = length

        # garbage collection to avoid using too much RAM
        del audio
        collect()

        print(
            f"created {len(final_segments)} {get_track_name(track, dir_path)} segments!\n"
        )


def start():
    print(
        r"""
_  _ _ ____ ___ ____ _  _   ___  ____ ____ ____ ____ ____ ____ ____ ____ 
|_/  | |__/  |  |__| |\ |   |__] |__/ |  | |    |___ [__  [__  |  | |__/ 
| \_ | |  \  |  |  | | \|   |    |  \ |__| |___ |___ ___] ___] |__| |  \ 
"""
    )
    print("starting kirtani processor")
    print("feel free to mess around, your original files will not be modified")
    print("recommendation: start with 1 track to find the right offsets for a samagam")
    print("press Ctrl-D to quit at any time")
    # --- CHOOSE DIR ---
    # for convenience during testing (CHECK_BEFORE_SUBMIT):
    # chosen_dir_path = "/Users/aman/Downloads/dodra"
    chosen_dir_path = fd.askdirectory(
        parent=root, initialdir=".", title="Please select a directory"
    )
    if chosen_dir_path[-1] == "/" or chosen_dir_path[-1] == "\\":
        chosen_dir_path = chosen_dir_path[:-1]
    if "edited" not in listdir(chosen_dir_path):
        mkdir(f"{chosen_dir_path}{slash(chosen_dir_path)}edited")
    print(f"\nyou chose {chosen_dir_path}")
    all_subdirs = tuple(walk(chosen_dir_path))

    SPEEDUP = True
    entered = input(
        """Enter Y if you have larger files to process and you would like to use more computing power
    [or press enter to go at the default accelerated, less stable speed]: """
    )
    if entered.lower() == "y":
        SPEEDUP = False

    # --- find mics used in files and set their relative offsets ---
    all_files = chain.from_iterable([filenames for _, _, filenames in all_subdirs])
    all_files = discard_other_files(all_files)
    mics = get_mic_names(all_files)
    print("found the following mics: ", list(mics.keys()))
    for mic in sorted(mics.keys()):
        while True:
            try:
                if mic in MIC_MAP:
                    entered = input(
                        f"Enter a DB value to offset by for mic {mic}, usually for {MIC_MAP[mic]} [or press enter to accept default value of {DB_MAP[MIC_MAP[mic]]}]: "
                    )
                    if entered.lower() == "n":
                        # don't normalize, just mix
                        mics[mic] = "n"
                        print(
                            f"mic {mic} will be mixed in with no normalization or offset\n"
                        )
                        break
                    db = int(entered) if entered else DB_MAP[MIC_MAP[mic]]
                else:
                    entered = input(
                        f"Enter a DB value to offset by for mic {mic} [or press enter to accept default value of -3]: "
                    )
                    if entered.lower() == "n":
                        # don't normalize, just mix
                        mics[mic] = "n"
                        print(
                            f"mic {mic} will be mixed in with no normalization or offset\n"
                        )
                        break
                    db = int(entered) if entered else -3
                print(f"    mic {mic} set to {db}\n")
                mics[mic] = db
                break
            except ValueError:
                print(
                    "Invalid input. Try again. If no normalization/offset is desired, enter the letter 'n'"
                )

    # use process-level parallelism across cores for CPU intensive tasks (editing tracks)
    with conc.ProcessPoolExecutor(
        max_workers=cpu_count() // 2 if SPEEDUP else 1
    ) as executor:
        futures = {
            executor.submit(edit_tracks, folder, chosen_dir_path, mics): folder
            for folder in all_subdirs
        }

        for future in conc.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                folder = futures[future]
                print(f"Error processing folder {folder}: {e}")


if __name__ == "__main__":
    startTime = time.time()
    start()

    # note that an increase of x decibels is 10^(x/10) fold increase
    executionTime = time.time() - startTime
    print("all done, look for a folder called 'edited'")
    print("execution time - " + format_time(round(executionTime) * 1000))
