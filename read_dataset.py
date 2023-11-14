from matplotlib.scale import scale_factory
import pyedflib
import numpy as np
import pandas as pd
import sys
import traceback
import random
import operator
import scipy
import os

def GetBatchIndexes(data_len, batch_num):
    batch_size = int(data_len / batch_num)
    idx_list = list(range(data_len))
    np.random.shuffle(idx_list)
    batch_idx_mask = []
    for i in range(batch_num):
        batch_idx_mask.append(idx_list[batch_size*i:batch_size*(i+1)])
    return batch_idx_mask

# 데이터 정리 
global state_list
state_list = ['ictal', 'preictal_late', 'preictal_early', 'preictal_ontime', 'postictal','interictal']
def LoadDataset(filename):
    df = pd.read_csv(filename)
    columns = ['name','start','end','state']
    interval_dict = {}
    for state in state_list:
        condition = df['state'] == state
        df_state = df[condition]
        interval_dict[state] = df_state[columns].values.tolist()
        
    return interval_dict

# state = ['ictal', 'preictal_late', 'preictal_early', 'preictal_ontime', 'postictal','interictal']
# output = [name, start, window_size]
def Interval2Segments(interval_list, data_path, window_size, sliding_size):
    segments_list = []
    for interval in interval_list:
        start = interval[1]
        end = interval[2]
        segment_num = int(((end-start-window_size)/sliding_size))+1
        for i in range(segment_num):
            segments_list.append([data_path+'/'+(interval[0].split('_'))[0]+'/'+interval[0]+'.edf', start, window_size])
            start += sliding_size

    return segments_list


def Interval2Segments_v2(interval_list, data_path, window_size, sliding_size):
    segments_list = []
    for interval in interval_list:
        name, start, end, state = interval[0], int(interval[1]), int(interval[2]), interval[3]
        segment_num = int(((end-start-window_size)/sliding_size))+1
        for i in range(segment_num):
            segments_list.append([os.path.join(data_path,(interval[0].split('_'))[0], name+'.edf'), start+sliding_size*i, window_size, state])
    return segments_list


def Segments2Data(segments):
    # segment[0] = 'filename', segment[1] = 'start', segment[2] = 'duration'
    channels = ['Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'Fp2-AVG',
                'F4-AVG', 'C4-AVG', 'P4-AVG', 'F7-AVG', 'T1-AVG',
                'T3-AVG', 'T5-AVG', 'O1-AVG', 'F8-AVG', 'T2-AVG',
                'T4-AVG', 'T6-AVG', 'O2-AVG', 'Fz-AVG', 'Cz-AVG',
                'Pz-AVG']
    channels_chb = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3',
                    'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                    'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8',
                    'P8-O2', 'FZ-CZ', 'CZ-PZ']
    signal_for_all_segments = []
    name = None
    read_end = 0
    f = None
    
    for idx, segment in enumerate(segments):
        if not name == segment[0]:
            name = segment[0]
            if not f == None:
                f.close()
            f = pyedflib.EdfReader(segment[0])
            skip_start = False   # 연속된 시간이면 한번에 읽기 위해 파일 읽는 시작 시간은 그대로 두고 끝 시간만 갱신함

        if not skip_start:
            interval_sets = [] # 연속된 구간이면 한번에 읽고 구간 정해진거에 따라 나누기 위해 구간 저장
            read_start = float(segment[1])
            read_end = 0
        # 최근 세그먼트의 start+window_size 값보다 read_end 값이 작으면 (읽는 끝값) read_end 값 갱신
        if read_end < float(segment[1]) + float(segment[2]):
            read_end = float(segment[1]) + float(segment[2])
        interval_sets.append([float(segment[1])-read_start, float(segment[1])+float(segment[2])-read_start, segment[3]])

        if not idx+1 >= len(segments) :
            # 파일이름이 같고, 다음 세그먼트의 시작시간이 더 크면서 현재 세그먼트의 시작시간 + window_size가 다음 세그먼트의 시작이랑 이어질 때
            if (name == segments[idx+1][0]) and (float(segment[1]) <= float(segments[idx+1][1])) and (float(segment[1]) + float(segment[2]) >= float(segments[idx+1][1])) :
                skip_start = True
                continue
        skip_start = False
                
        freq = f.getSampleFrequencies()
        labels = f.getSignalLabels()
        chn_num = len(channels)

        # UpSampling을 위해 x 값 생성
        x = np.linspace(0, 10,int((read_end-read_start)*freq[0]) )
        x_upsample = np.linspace(0,10,int(256*(read_end-read_start)))

        seg = []
        y = []
        for i in range(len(interval_sets)):
            seg.append([])

        target_freq =120;
        if 'SNU' in name:
            for channel in channels:
                ch_idx = labels.index(channel)
                edf_signal = f.readSignal(ch_idx,int(freq[ch_idx]*read_start),int(freq[ch_idx]*(read_end-read_start)))
                
                # 256 Hz이하일 경우 256Hz로 interpolation을 이용한 upsampling
                
                if not freq[ch_idx] == target_freq:
                    signal = np.interp(x_upsample,x, edf_signal)
                else:
                    signal = edf_signal
                
                for j in range(len(interval_sets)):
                    
                    seg[j].append( list(signal[int(interval_sets[j][0] * target_freq) : int(interval_sets[j][1] * target_freq) ]) )
        
        if 'CHB' in name:            
            for channel in channels_chb:
                ch_idx = labels.index(channel)
                edf_signal = f.readSignal(ch_idx,int(freq[ch_idx]*read_start),int(freq[ch_idx]*(read_end-read_start)))
                # 256 Hz이하일 경우 256Hz로 interpolation을 이용한 upsampling
                if not freq[ch_idx] == target_freq:
                    signal = np.interp(x_upsample,x, edf_signal)
                else:
                    signal = edf_signal
                
                for j in range(len(interval_sets)):
                    seg[j].append( list(signal[int(interval_sets[j][0] * target_freq) : int(interval_sets[j][1] * target_freq) ]) )
            for s in seg:    
                signal_for_all_segments.append(s)

        skip_start = False
            
    if hasattr(f,'close'):
         f.close()

    return np.array(signal_for_all_segments)/10


def Segments2Data_v2(segments):
    # segment[0] = 'filename',
    # segment[1] = 'start',
    # segment[2] = 'duration'
    # segment[3] = 'state'
    channels_dict = \
        {
            "SNU": ['Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'Fp2-AVG',
                    'F4-AVG', 'C4-AVG', 'P4-AVG', 'F7-AVG', 'T1-AVG',
                    'T3-AVG', 'T5-AVG', 'O1-AVG', 'F8-AVG', 'T2-AVG',
                    'T4-AVG', 'T6-AVG', 'O2-AVG', 'Fz-AVG', 'Cz-AVG',
                    'Pz-AVG'],
            "CHB": ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3',
                    'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                    'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8',
                    'P8-O2', 'FZ-CZ', 'CZ-PZ']
        }
        
    x = [] # return value
    y = [] # return value
    
    # combine segments on same file to read once.
    segments_on_same_file_dict = dict()
    
    for segment in segments:
        file, start, duration, state = segment
        try:
            segments_on_same_file_dict[file].append([start, duration, state])
        except:
            segments_on_same_file_dict[file] = [[start, duration, state]]
            
    for file, segment_list in segments_on_same_file_dict.items():
        if "CHB" in file:
            used_channels = channels_dict["CHB"]
        elif "SNU" in file:
            used_channels = channels_dict["SNU"]
        else: continue
        
        with pyedflib.EdfReader(file) as edf_reader:
            edf_labels = edf_reader.getSignalLabels()
            edf_freq = list(map(int, edf_reader.getSampleFrequencies()))

            if not all([(channel in edf_labels) for channel in used_channels]): continue
                
            for segment_info in segment_list:
                start, duration, state = segment_info
                
                segment_data = []
                scale_factor = 10
                target_freq = 128
                
                for channel in used_channels:
                    channel_index = edf_labels.index(channel)
                    channel_freq = edf_freq[channel_index]
                    start_index = int(start)*channel_freq
                    duration_length = int(duration)*channel_freq

                    signal = edf_reader.readSignal(channel_index,start_index, duration_length)
                    
                    resample_len = int(len(signal)*target_freq/channel_freq)
                    if channel_freq!=target_freq:
                        signal = scipy.signal.resample(signal, resample_len)
                        
                    scaled_signal = signal/scale_factor
                    segment_data.append(scaled_signal)

                x.append(segment_data)
                
                if state == 'preictal_ontime':
                    y.append(1)
                else:
                    y.append(0)
    return np.array(x), np.array(y)
