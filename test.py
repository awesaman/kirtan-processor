from pydub import AudioSegment, silence, effects
import numpy as np

kirtani_track = '/Users/aman/Downloads/raw_mp3_SatEve/ZOOM0010_unsegmented.mp3'
kirtani = AudioSegment.from_file(kirtani_track)
# kirtani = effects.normalize(kirtani) - 3

dBFS=kirtani.dBFS

segments = silence.detect_nonsilent(kirtani, min_silence_len=3000, silence_thresh=dBFS-22, seek_step=1000)
final_cuts = []
prev_end = 0
min_time_between_vaaris = 10000
dropout = 60000
for start,end in segments:
    # drop out anything less than 60 seconds
    if end - start < dropout:
        continue

    # the longer the previous segment is, the less likely we are to merge the current segment into it
    if prev_end:
        prev_length = final_cuts[-1][1]-final_cuts[-1][0]
        
        # case if prev segment is less than 16 minutes
        if prev_length < 16*60000 or start - prev_end < min_time_between_vaaris:
            final_cuts[-1][1] = end
        else:
            final_cuts.append([start,end])
    # case if first segment of recording
    
    else:
        final_cuts.append([start,end])
    prev_end = end
    print('{}:{} - {}:{}'.format(start//60000,(start//1000)%60,end//60000,(end//1000)%60))
print(['{}:{} - {}:{}'.format(start//60000,(start//1000)%60,end//60000,(end//1000)%60) for start,end in final_cuts])

'''
0:00 - 41:15
59:46 - 60:12
60:28 - 185:18
185:37 - 204:15
204:31 - 229:30
229:43 - 235:09
235:19 - 256:51
257:01 - 288:30
'''