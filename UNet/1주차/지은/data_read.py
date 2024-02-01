import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as pyplot

#data 불러오기
#ISBI 2012 EM Segmentation Challenge에 사용된 membrane 데이터셋
dir_data = './datasets'

name_label = 'train-labels.tif' #512x512x30 
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data,name_label))
img_input = Image.open(os.path.join(dir_data,name_input))

#데이터 전처리
ny,nx = img_label.size #512,512
nframe = img_label.n_frames #30 

#학습, 훈련, 검증 데이터를 24, 3, 3개씩 분류 후, train, test, val에 저장
nframe_train=24
nframe_val=3
nframe_test=3

#데이터 저장 폴더 구조 생성
dir_save_train = os.path.join(dir_data,'train')
dir_save_val = os.path.join(dir_data,'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

#랜덤하게 이미지 추출
id_frame = np.arange(nframe)
np.random.shuffle(id_frame) 

#train data
#id_frame에서 0번재 id부터 시작
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i+offset_nframe])
    img_input.seek(id_frame[i+offset_nframe])
    
    label_=np.asarray(img_label)
    input_=np.asarray(img_input)
    
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

#val data 
offset_nframe += nframe_train #24번째 아이디부터 시작

for i in range(nframe_val): #3번 반복
    img_label.seek(id_frame[i+offset_nframe]) #0+24, 1+24, 2+24 -> 24,25,26 data 들어감 
    img_input.seek(id_frame[i+offset_nframe])
    
    #array(복사 후 원본을 변경할 경우, 업데이트 해도 변경되지 않음) vs asarray(복사 후, 업데이트 하면 원본이 변경됨)
    # https://ok-lab.tistory.com/179
    label_=np.asarray(img_label)
    input_=np.asarray(img_input)
    
    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

#test data    
offset_nframe += nframe_val #26번째 아이디부터 시작

for i in range(nframe_test): #3번 반복
    img_label.seek(id_frame[i+offset_nframe]) 
    img_input.seek(id_frame[i+offset_nframe])
    
    label_=np.asarray(img_label)
    input_=np.asarray(img_input)
    
    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)