import os
import natsort
from pyedflib.highlevel import read_edf, read_edf_header
import pandas as pd
import csv
from datetime import datetime, timedelta

ictal_section_name = ['ictal', 'preictal_late', 'preictal_ontime', 'preictal_early', 'postictal', 'interictal']

SOP = 30 #minutes
SPH = 2 #minutes
PREICTAL_EARLY_DURATION = 60 # minutes
POSTICTAL_DURATION = 120 # minutes

def get_summary_info(path: str, file: str)-> list:
    """_summary_

    Args:
        path (str): directory path
        file (str): name of file as "file.txt"

    Returns:
        list: it contains seizure_file dictionary.
            seizure_file = {'name': filename,
                            'seizure': [[start, end], [start, end], ...]}
    """
    with open(os.path.join(path, file), 'r') as f:        
        contents = f.read().split("\n\n")
        seizure_info_list = []

        for edf_info in contents:
            try:
                info = edf_info.split("\n")
                
                for idx, line in enumerate(info):
                    if ('Name' in line) or ('name' in line):
                        file_name_idx = idx
                    
                    if ('Number' in line) or ('number' in line):
                        number_of_seizures_idx = idx
                
                num_of_seizure = int(list(info[number_of_seizures_idx].split(": "))[-1])
                
                if num_of_seizure>0:
                    filename = list(info[file_name_idx].split(': '))[-1]
                    seizure_file = {'name': filename,
                                    'seizure': []}

                    for i in range(num_of_seizure):
                        seizure_start_idx = number_of_seizures_idx + 2*i
                        seizure_end_idx = number_of_seizures_idx + 2*i + 1
                        seizure_start = int(info[seizure_start_idx].split(": ")[-1].rstrip(' seconds'))
                        seizure_end = int(info[seizure_end_idx].split(": ")[-1].rstrip(' seconds'))
                        
                        seizure_file['seizure'].append([seizure_start, seizure_end])

                    seizure_info_list.append(seizure_file)       
            except:
                pass

        return seizure_info_list            

data_path = f".\\data\\chb"

file_list = natsort.natsorted(os.listdir(data_path)) # 이름순으로 순서 정렬
patient_folder_list = []

for file in file_list:
	file_path = os.path.join(data_path, file)
	if os.path.isdir(file_path):
		patient_folder_list.append([file_path, file])
		# [0] : 폴더 경로
		# [1] : 폴더 내 파일 이름 ex) chb01

# edf_header.keys() = dict_keys(['technician', 'recording_additional', 'patientname', 'patient_additional', 'patientcode', 'equipment', 'admincode', 'sex', 'startdate', 'birthdate', 'gender', 'Duration', 'SignalHeaders', 'channels', 'annotations'])
 
disordered_seizure_info_list = []

for path_and_patient in patient_folder_list:
    path, patient = path_and_patient

    edf_list = natsort.natsorted([file for file in os.listdir(path)
                                  if ("seizure" not in file)
                                  and ("summary" not in file)])
    
    first_edf, last_edf = edf_list[0], edf_list[-1]
    
    first_edf_header = read_edf_header(os.path.join(path, first_edf))
    startdate = first_edf_header['startdate'] # datetime.datetime()

    last_edf_header = read_edf_header(os.path.join(path, last_edf))
    enddate = last_edf_header['startdate'] + timedelta(seconds=last_edf_header['Duration']) # datetime.datetime()
    
    current_seizure_info = {'name': patient,
                            'startdate': startdate,
                            'enddate': enddate,
                            'ictal': list(),
                            'preictal_late': list(),
                            'preictal_ontime': list(),
                            'preictal_early': list(),
                            'postictal': list(),
                            'interictal': list()}

    summary = patient+'-summary.txt'
    summary_info_list = get_summary_info(path, summary)
    
    for summary_info in summary_info_list:
        edf_header = read_edf_header(os.path.join(path, summary_info['name']))
        file_startdate = edf_header['startdate']

        for start_sec,end_sec in summary_info['seizure']:
            
            ictal_start = file_startdate + timedelta(seconds=start_sec)
            ictal_end = file_startdate + timedelta(seconds=end_sec)
            
            preictal_late_start = ictal_start - timedelta(minutes=SPH)
            preictal_late_end = ictal_start
            
            preictal_ontime_start = preictal_late_start - timedelta(minutes=SOP)
            preictal_ontime_end = preictal_late_start
            
            preictal_early_start = ictal_start - timedelta(minutes=PREICTAL_EARLY_DURATION)
            preictal_early_end = preictal_ontime_start
            
            postictal_start = ictal_end
            postictal_end = ictal_end + timedelta(minutes=POSTICTAL_DURATION)
            
            current_seizure_info['ictal'].append([ictal_start, ictal_end])
            current_seizure_info['preictal_late'].append([preictal_late_start, preictal_late_end])
            current_seizure_info['preictal_ontime'].append([preictal_ontime_start, preictal_ontime_end])
            current_seizure_info['preictal_early'].append([preictal_early_start, preictal_early_end]) 
            current_seizure_info['postictal'].append([postictal_start, postictal_end])
            
    disordered_seizure_info_list.append(current_seizure_info)

ordered_seizure_info_list = list()

for current_seizure_info in disordered_seizure_info_list:
    name = current_seizure_info['name']
    startdate = current_seizure_info['startdate']
    enddate = current_seizure_info['enddate']
    
    # ictal_section[i][0] == start datetime of i th ictal_section
    # ictal_section[i][1] == end datetime of i th ictal_section    
    ictal = current_seizure_info['ictal']
    preictal_late = current_seizure_info['preictal_late']
    preictal_ontime = current_seizure_info['preictal_ontime']
    preictal_early = current_seizure_info['preictal_early']
    postictal = current_seizure_info['postictal']
    
    temp_preictal_late_list = []
    temp_preictal_ontime_list = []
    temp_preictal_early_list = []
    temp_postictal_list = []
    temp_interictal_list = []
    
    for i in range(len(ictal)):    
        if i==0: # first section(near startdate)
            if startdate<preictal_late[i][1]:
                temp_preictal_late_list.append([max(startdate, preictal_late[i][0]), preictal_late[i][1]])
              
            if startdate<preictal_ontime[i][1]:
                temp_preictal_ontime_list.append([max(startdate, preictal_ontime[i][0]), preictal_ontime[i][1]])
              
            if startdate<preictal_early[i][1]:
                temp_preictal_early_list.append([max(startdate, preictal_early[i][0]), preictal_early[i][1]])
              
            if startdate<preictal_early[i][0]:
                temp_interictal_list.append([startdate, preictal_early[i][0]])
               
            if postictal[i][0]<preictal_early[i+1][0]:
                temp_postictal_list.append([postictal[i][0], min(postictal[i][1], preictal_early[i+1][0])])
                
            if postictal[i][1]<preictal_early[i+1][0]:
                temp_interictal_list.append([postictal[i][1], preictal_early[i+1][0]])
               
            continue
    
        if i==(len(ictal)-1): # last section(near enddate)
            if ictal[i-1][1]<preictal_late[i][1]:
                temp_preictal_late_list.append([max(ictal[i-1][1], preictal_late[i][0]), preictal_late[i][1]])
            
            if ictal[i-1][1]<preictal_ontime[i][1]:
                temp_preictal_ontime_list.append([max(ictal[i-1][1], preictal_ontime[i][0]), preictal_ontime[i][1]])
                
            if ictal[i-1][1]<preictal_early[i][1]:
                temp_preictal_early_list.append([max(ictal[i-1][1], preictal_early[i][0]), preictal_early[i][1]])
                
            if postictal[i][0]<enddate:
                temp_postictal_list.append([postictal[i][0], min(postictal[i][1], enddate)])
                
            if postictal[i][1]<enddate:
                temp_interictal_list.append([postictal[i][1], enddate])
                
            continue

        # middle section
        if ictal[i-1][1]<preictal_late[i][1]:
            temp_preictal_late_list.append([max(ictal[i-1][1], preictal_late[i][0]), preictal_late[i][1]])
            
        if ictal[i-1][1]<preictal_ontime[i][1]:
            temp_preictal_ontime_list.append([max(ictal[i-1][1], preictal_ontime[i][0]), preictal_ontime[i][1]])
            
        if ictal[i-1][1]<preictal_early[i][1]:
            temp_preictal_early_list.append([max(ictal[i-1][1], preictal_early[i][0]), preictal_early[i][1]])
            
        if postictal[i][0]<preictal_early[i+1][0]:
            temp_postictal_list.append([postictal[i][0], min(postictal[i][1], preictal_early[i+1][0])])
            
        if postictal[i][1]<preictal_early[i+1][0]:
            temp_interictal_list.append([postictal[i][1], preictal_early[i+1][0]])
            
    ordered_current_seizure_info = {'name': name,
                                    'startdate': startdate,
                                    'enddate': enddate,
                                    'ictal': ictal,
                                    'preictal_late': temp_preictal_late_list,
                                    'preictal_ontime': temp_preictal_ontime_list,
                                    'preictal_early': temp_preictal_early_list,
                                    'postictal': temp_postictal_list,
                                    'interictal': temp_interictal_list}
    
    ordered_seizure_info_list.append(ordered_current_seizure_info)



patient_segment_list = []
columns=['name','start','end','state']

for current_patient in ordered_seizure_info_list:
    name = current_patient['name']
    name = name[:3].upper()+'0'+name[3:]
    for state in ictal_section_name:
        for start, end in current_patient[state]:
            patient_segment_list.append([name, start, end, state])

df = pd.DataFrame(patient_segment_list, columns=['name','start','end','state'])
df.to_csv('./patient_info_chb.csv',index=False)