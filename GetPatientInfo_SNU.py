import os
import natsort
import pyedflib
import pandas as pd
import csv

data_path = "C:\\Users\\jangdabin\\Desktop\\Epilepsy\\Code\\data\\snu"

file_list = natsort.natsorted(os.listdir(data_path)) # 이름순으로 순서 정렬
patient_folder_list = []

for file in file_list:
	file_path = data_path + '/' + file
	if os.path.isdir(file_path):
		patient_folder_list.append([file_path,file])
# [0] : 폴더 경로 	[1] : 폴더 내의 .txt파일 이름 ex) patient_1

total_seizure_info_list = []

#{'name':환자 이름, 'ictal':[ [start,end], [start,end], ... ] } 형태의 환자별 딕셔너리로 만든 후 리스트에 추가
#{'EndTime' : Value } 는 EDF에서 파일 읽어서 끝나는 시간 계산
for patient in patient_folder_list:
	temp_set = {'name': patient[1]}
	with open(patient[0]+'/'+patient[1]+'.txt','r') as f:
		temp_list = []
		for line in f:
			time = line.strip()
			seizure_time_set = (time.split(','))	# 쉼표로 시작 끝시간 구분
			temp_list.append([int(seizure_time_set[0]), int(seizure_time_set[1])])	
		temp_set['ictal'] = temp_list
	
	with pyedflib.EdfReader(patient[0]+'/'+patient[1]+'.edf') as f:
		duration = f.getFileDuration()
		temp_set['endtime'] = duration
	total_seizure_info_list.append(temp_set)


SOP = 30
SPH = 2
for info in total_seizure_info_list:
	info['ictal'].sort(key= (lambda x:x[0]) )	# 시작 시간 순서대로 정렬
	seizure_time_set = info['ictal']
	interictal_list = []
	preictal_1hour_list = []
	preictal_list = []
	preictal_late_list = []
	post_ictal_list = []
	for i in range(len(seizure_time_set)):
		seizure_start_time = seizure_time_set[i][0]
		seizure_end_time = seizure_time_set[i][1]
		# 현재 시간 기준으로 inter-ictal 추출
		### inter-ictal ###
		if i == 0:
            # ictal 1시간 전이 0보다 작으면 ictal 앞부분에는 interictal 없음
			if not seizure_start_time - 3600 < 0:      
				interictal_list.append([0,seizure_start_time-3600])
		else:
			if not seizure_start_time - 3600 < seizure_time_set[i-1][1]+7200 : # Ictal - 1h 가 그 전 Ictal이 끝나고 PostIctal 구간일 때 제외
				# ictal이 지나고 2시간 뒤부터 현재 ictal 1시간 전까지
				interictal_list.append([seizure_time_set[i-1][1]+7200, seizure_start_time-3600])
		if i == len(seizure_time_set)-1:	# 마지막 seizure일 경우 endtime 사이에서의 interictal 계산산
			if not seizure_end_time + 7200 > info['endtime']:
				interictal_list.append([ seizure_end_time+7200, info['endtime'] ]) 

		###  pre-ictal 1-hour  ###
		
		# 첫 ictal일 경우 preictal_1hour(32min ~ 60min) 시간 계산 시 0보다 작은 값 나오지 않도록 처리리
		if i == 0:
			# seizure_start_time - (SOP+SPH)*60  < 0 인 경우 preictal-1hour 없음
			if not seizure_start_time - (SOP+SPH)*60 < 0: # (SOP+SPH)*60 = 1920   
				preictal_1hour_end = seizure_start_time - (SOP+SPH)*60
				if seizure_start_time - 3600 < 0: # 계산된 (ictal - 1hour) preictal-1hour의 시작시간이 0보다 작으면 시작시간 0으로
					preictal_1hour_start = 0
				else:
					preictal_1hour_start = seizure_start_time - 3600 # 아닐 경우 ictal - 1hour 로 시작시간 설정정
				
				preictal_1hour_list.append( [preictal_1hour_start, preictal_1hour_end] )
		else:
			# postictal이 끝나는 시간이 전의 seizure_endtime보다 늦을경우 preictal-1hour 없음
			if not seizure_time_set[i-1][1] >  seizure_start_time - (SOP+SPH)*60:
				preictal_1hour_end = seizure_start_time - (SOP+SPH)*60
				if seizure_start_time - 3600 < seizure_time_set[i-1][1]:
					# seizure_end_time이 끝나는 시간이 preictal-1hour 구간 사이에 걸쳐있을 경우 postictal이 끝나는 시점을 preictal-1hour 구간의 시작으로 설정
					preictal_1hour_start = seizure_time_set[i-1][1]
				else:
					preictal_1hour_start = seizure_start_time - 3600
				
				preictal_1hour_list.append( [preictal_1hour_start, preictal_1hour_end] )

		###  SOP + SPH (preictal)  ###
		if i == 0:
			# ictal 시작시간 - SPH(2분)가 0보다 작을 경우 preictal 없음
			if not seizure_start_time - SPH * 60 < 0 :
				preictal_end = seizure_start_time - SPH * 60
				# preictal의 시작시간이 0보다 작지 않도록 예외 처리
				if seizure_start_time - (SOP + SPH)*60 < 0: 
					preictal_start = 0
				else:
					preictal_start = seizure_start_time - (SOP + SPH)*60 
				
				preictal_list.append( [preictal_start, preictal_end] )
		else:
			# 이전 seizure시작시간 - SPH(2 min)이 이전 seizure가 끝나기 전이면 스킵
			if not seizure_time_set[i-1][1] >  seizure_start_time - SPH*60:
				preictal_end = seizure_start_time - SPH*60
				if seizure_start_time - (SOP + SPH)*60 < seizure_time_set[i-1][1]:
					preictal_start = seizure_time_set[i-1][1]
				else:
					preictal_start = seizure_start_time - (SOP + SPH)*60
				
				preictal_list.append( [preictal_start, preictal_end] )
		
		if i == 0:
			if seizure_start_time - SPH * 60 < 0:
				preictal_late_start = 0
			else:
				preictal_late_start = seizure_start_time - SPH*60
			preictal_late_end = seizure_end_time
			preictal_late_list.append( [preictal_late_start, preictal_late_end] )
		else:
			if seizure_start_time - SPH * 60 < seizure_time_set[i-1][1]:
				preictal_late_start = seizure_time_set[i-1][1]
			else:
				preictal_late_start = seizure_start_time - SPH*60
			preictal_late_end = seizure_end_time
			preictal_late_list.append( [preictal_late_start, preictal_late_end] )




		### Post ictal ###
		if i == len(seizure_time_set)-1:
			postictal_start = seizure_end_time
			if seizure_end_time + 7200 > info['endtime']:
				postictal_end = info['endtime']
			else:
				postictal_end = seizure_end_time + 7200
			
			post_ictal_list.append( [postictal_start, postictal_end] )
			
		else:
			if not seizure_time_set[i+1][0]-3600 < seizure_end_time :
				postictal_start = seizure_end_time
				if seizure_time_set[i+1][0]-3600 < seizure_end_time + 7200:
					postictal_end = seizure_time_set[i+1][0]-3600
				else:
					postictal_end = seizure_end_time + 7200
				post_ictal_list.append( [postictal_start, postictal_end] )

				
	info['ictal'] = seizure_time_set
	
	if interictal_list: # interictal 리스트가 비어있지 않으면
		info['interictal'] = interictal_list
	else:
		info['interictal'] = None

	if preictal_1hour_list: # preictal-1hour리스트가 비어있지 않으면
		info['preictal_early'] = preictal_1hour_list
	else:
		info['preictal_early'] = None
	
	if preictal_list: # preictal 리스트가 비어있지 않으면
		info['preictal_ontime'] = preictal_list
	else:
		info['preictal_ontime'] = None
	
	if preictal_late_list:
		info['preictal_late'] = preictal_late_list
	else:
		info['preictal_late'] = None
	
	if post_ictal_list: # postictal 리스트가 비어있지 않으면
		info['postictal'] = post_ictal_list
	else:
		info['postictal'] = None


patient_segments_list = []
for patient in total_seizure_info_list:
	patient_number = int((patient['name'].split('_'))[1])
	patient_name_snu = "SNU%03d"%patient_number
	dict_keys = list(patient.keys())
	for i in range(len(dict_keys)):
		if dict_keys[i] == 'name' or dict_keys[i] == 'endtime':
			continue
		time_list = patient[dict_keys[i]]
		if not time_list==None:
			for time in time_list:
				patient_segments_list.append([patient_name_snu, time[0], time[1], dict_keys[i]])



df = pd.DataFrame(patient_segments_list,columns=['name','start','end','state'])
df.to_csv('./patient_info.csv',index=False)












