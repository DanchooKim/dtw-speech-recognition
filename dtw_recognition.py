import librosa
import tslearn
from tslearn.metrics import dtw,dtw_path
import numpy as np
import pathlib
import sys
import re
import random
import webrtcvad
import sounddevice as sd
import copy
from tqdm import tqdm
from scipy.io import wavfile

vad = webrtcvad.Vad()
vad.set_mode(3)

#addable_pattern = ''

# 사용할 단어들 
google_word =['bed','bird','cat','dog','down','eight','five','four','go','happy','house','left','marvin','nine','no','off','on','one','right','seven','sheila','six','stop','three','tree','two','up','wow','yes','zero']

WORDS = google_word
# [(word1,mfccs1),(word2,mfccs2).................]
CODEBOOK = []

'''####################
END_POINT_DETECTION
####################'''

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    frames = []
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        frames.append(Frame(audio[offset:offset + n], timestamp, duration))
        timestamp += duration
        offset += n   
    return frames

def auto_vad(vad, samples, sample_rate, frame_duration_ms = 10):
    speech_frame = []
    frames = frame_generator(frame_duration_ms, samples, sample_rate)
    #print(n_frame)
    for idx, frame in enumerate(frames):
        if vad.is_speech(frame.bytes, sample_rate):
            speech_frame.append(idx)

    cutted_samples = []
    
    for i in speech_frame:
        b = frames[i].bytes
        cutted_samples.extend(np.frombuffer(b, dtype=np.int16))

    return cutted_samples

'''####################
READ_FILE
####################'''
def remove_np_file(absolute_super_dir_path):
    import os
    from pathlib import Path

    path = list(Path(absolute_super_dir_path).rglob("*.npy"))
    for i in tqdm(path,desc = "remove!!"):
        os.remove(i)
    
def load_filepath_for_speech_commands(absolute_txt_path,addable_path):
    file_list_tuple = set()
    try:
        file = open(absolute_txt_path, 'r', encoding = "utf-8")
        raw_list = file.readlines()
    except:
        print('[Error]No such file:%s'%absolute_txt_path)
        return file_list_tuple

    raw_list = [re.sub(addable_path,"",i) for i in raw_list]
    raw_list = [re.sub("\n","",i) for i in raw_list]
    pattern = re.compile(r"/.+")
    file_list_tuple= [(pattern.sub("",i),addable_path+i) for i in raw_list]
    return file_list_tuple

def mfcc_from_wavpath(absolute_file_path):
    wav, _ = librosa.load(absolute_file_path, sr =16000)
    mfcc = librosa.feature.mfcc(y=wav, sr = 16000, n_mfcc=13, hop_length = 160)
    return mfcc.T

#VAD 과정이 소스에 포함되어 있음
def read_trimmed_mfcc_from_wavpath(absolute_file_path):
    import os
    np_path = re.sub(".wav",".npy",absolute_file_path)
    
    if os.path.isfile(np_path):
        mfccT = np.load(np_path)
        return mfccT


    sr,wav= wavfile.read(absolute_file_path)
    original_wav = wav

    if sr != 16000:
        wav,_ = librosa.load(absolute_file_path,sr=sr)
        resample = librosa.resample(wav,sr,16000)
        resample = np.float32(resample)
        resample =resample*2**15
        resample =np.int16(resample)
        wav = resample
    
    new_wav = auto_vad(vad, wav, sample_rate=16000, frame_duration_ms=10)
    new_wav = np.int16(new_wav)
    new_wav = np.float32((new_wav)/2**15)

    try:
        mfcc = librosa.feature.mfcc(y=new_wav, sr = 16000, n_mfcc=13, hop_length=160)
        mfccT = mfcc.T
        np.save(re.sub(".npy","",np_path), mfccT,allow_pickle=False)
    except:
        new_wav =  np.float32((original_wav)/2**15)
        mfcc = librosa.feature.mfcc(y=new_wav, sr = 16000, n_mfcc=13, hop_length=160)
        mfccT = mfcc.T
        np.save(re.sub(".npy","",np_path), mfccT,allow_pickle=False)
    return mfccT

'''
def read_trimmed_mfcc_from_wavpath(absoulte_file_path):
    wav,_ = librosa.load(absoulte_file_path, sr=48000)
    mfcc = librosa.feature.mfcc(y=wav, sr=48000)
    return mfcc
'''
def play_timmed_audio_from_wavpath(absolute_file_path):
    sr,wav= wavfile.read(absolute_file_path)
    
    if sr != 16000:
        wav,_ = librosa.load(absolute_file_path,sr=sr)
        resample = librosa.resample(wav,sr,16000)
        resample = np.float32(resample)
        resample =resample*2**15
        resample =np.int16(resample)
        wav = resample
    new_wav = auto_vad(vad, wav, sample_rate=16000, frame_duration_ms=10)
    new_wav = np.int16(new_wav)
    new_wav = np.float32((new_wav)/2**15)
    sd.play(new_wav,samplerate=16000)

    return


'''####################
MODEL
####################'''

def guess_tuple(absolute_file_path):
    mfcc = read_trimmed_mfcc_from_wavpath(absolute_file_path)
    min_tuple = tuple()
    min = sys.maxsize
    for ref_tuple in CODEBOOK:
        path, tmp= tslearn.metrics.dtw_path(ref_tuple[1],mfcc)
        if tmp< min:
            min = tmp
            min_tuple = ref_tuple
    return min_tuple[0]

def guess_tuple_novad(absolute_file_path):
    mfcc = mfcc_from_wavpath(absolute_file_path)
    min_tuple = tuple()
    min = sys.maxsize
    for ref_tuple in CODEBOOK:
        path, tmp= tslearn.metrics.dtw_path(ref_tuple[1],mfcc)
        if tmp< min:
            min = tmp
            min_tuple = ref_tuple
    return min_tuple[0]

def get_list_number(list_length,amount_of_data,order='random'):
    if order == 'fix':
        return list(range(0,amount_of_data))
    else:
        num_list = []
        ran_num = random.randint(0,list_length-1)
        for i in range(amount_of_data):
            while ran_num in num_list:
                ran_num = random.randint(0,list_length-1)
            num_list.append(ran_num)
        num_list.sort()
        return num_list

def set_codebook_normal(codebook_size_of_each_word,order ='fix'):
    '''
    if order == fix, then codebook of each word will be each_word_path[0:codebook_size_of_each_word]
    if order == random, then codebook of each word will be random 
    '''
    global CODEBOOK
    for i in WORDS:
        path = list(pathlib.Path('C:/Users/김남형/Desktop/연구/연구소스/data/speech_commands_v0.01/'+i+'/').glob('*.wav'))
        number_list = get_list_number(list_length=len(path),amount_of_data=codebook_size_of_each_word, order = order)
        #print(number_list)
        for j in number_list:
            #print(len(path))
            try :
                CODEBOOK.append((i,read_trimmed_mfcc_from_wavpath(path[j].as_posix())))
            except :
                new_j = random.randint(0,len(path)-1)    
                while new_j in number_list:
                    new_j = random.randint(0,len(path)-1)
                number_list.append(new_j)
    return
def set_codebook_with_dtw_k_means(codebook_size_of_each_word, itera = 10):
    global CODEBOOK
    mfccs_dict = dict() # bed:[(0, mfccs)..................], qu:[(0, mfccs)..................]
    mfccs_cluster_dict = dict()
    code_book_for_kmeans = [] # (k,WORD,mfccs)
    
    #mfcc 다불러와서 여기 저장

    #형태:BED: [(0,mfcc),(0,mfcc)...... (0,mfcc)], CAT: [(0,mfcc),(0,mfcc)...... (0,mfcc)]
    for i in tqdm(WORDS, desc="Extract_MFCC"):
        path =list(pathlib.Path('C:/Users/김남형/Desktop/연구/연구소스/data/speech_commands_v0.01/'+i+'/').glob('*.wav'))
        mfccs_set = []
        for j  in path:
            mfccs_set.append((0,read_trimmed_mfcc_from_wavpath(j.as_posix())))
        mfccs_dict[i] = copy.deepcopy(mfccs_set)

    #각 word의 initial 센트로이드 설정
    for i in tqdm(WORDS, desc = "initialize_centroid"):
        mfccs_set = []
        for j in range(codebook_size_of_each_word):
            word_mfccs_set = copy.deepcopy(mfccs_dict[i])
            mfccs_set.append((j,copy.deepcopy(word_mfccs_set[j][1])))
        mfccs_cluster_dict[i] = copy.deepcopy(mfccs_set)

    # mfccs_cluster_dict : initial 센트로이드들 들어있음.
    for loop in tqdm(range(itera)):     
    #k means 시작 지금은 루프를 씌워서 작동함.
        for key in WORDS:
        # Word 별로 kmeans 하므로 가져온다.
            key_mfccs_dict = copy.deepcopy(mfccs_dict[key]) #type = tuples array
            key_mfccs_cluster_dict = copy.deepcopy(mfccs_cluster_dict[key])#type =tuples array
            processed_key_mfccs_dict = list()

            for mth_key_mfccs_dict in key_mfccs_dict:
                tmp=0 # k번 반복한 후에 다시 0으로 바뀌기 때문에 이거 하나면 된다
                for kth_key_mfccs_cluster_dict in key_mfccs_cluster_dict:
                
                    path_of_kmth, distance_of_kmth =dtw_path(kth_key_mfccs_cluster_dict[1],mth_key_mfccs_dict[1])
                    if tmp == 0:
                        selected_path_of_mth = copy.deepcopy(path_of_kmth)
                        selected_distance_of_mth = distance_of_kmth
                        selected_cluster_of_mth = kth_key_mfccs_cluster_dict[0]
                        tmp = 1
                        continue 
                    if distance_of_kmth <= selected_distance_of_mth:
                        selected_path_of_mth = copy.deepcopy(path_of_kmth)
                        selected_distance_of_mth = distance_of_kmth
                        selected_cluster_of_mth = kth_key_mfccs_cluster_dict[0]
            
                processed_mfccs= make_ylength_to_xlength_with_dtw(key_mfccs_cluster_dict[selected_cluster_of_mth][1],mth_key_mfccs_dict[1],selected_path_of_mth)
                processed_key_mfccs_dict.append((selected_cluster_of_mth,copy.deepcopy(processed_mfccs)))
        
            number_of_clusters_sample = [1 for i in range(codebook_size_of_each_word)]

            #각 클러스터의 갯수를 세야된다. 그다음에 key_mfccs_cluster_dict에 더해줘야된다.
            for i in processed_key_mfccs_dict:
                number_of_clusters_sample[i[0]] += 1
                key_mfccs_cluster_dict[i[0]] = copy.deepcopy((i[0],copy.deepcopy(np.add(i[1],key_mfccs_cluster_dict[i[0]][1]))))
        
            #이제 클러스터 갯수만큼 나눠준다
            for alpha in key_mfccs_cluster_dict:
                key_mfccs_cluster_dict[alpha[0]]= copy.deepcopy((alpha[0],alpha[1]/number_of_clusters_sample[alpha[0]]))
        
            #원본에다가 저장해야된다.
            mfccs_cluster_dict[key]=copy.deepcopy(key_mfccs_cluster_dict)
    #변환
    for key in WORDS:
        dumy = [CODEBOOK.append((key,i[1])) for i in mfccs_cluster_dict[key]]
        #저장을 안할거지만 list comprehension 때문에 생성한 변수임 dummy는.

def set_codebook_with_dtw_k_means_novad(codebook_size_of_each_word, itera = 10):
    global CODEBOOK
    mfccs_dict = dict() # bed:[(0, mfccs)..................], qu:[(0, mfccs)..................]
    mfccs_cluster_dict = dict()
    code_book_for_kmeans = [] # (k,WORD,mfccs)
    
    #mfcc 다불러와서 여기 저장

    #형태:BED: [(0,mfcc),(0,mfcc)...... (0,mfcc)], CAT: [(0,mfcc),(0,mfcc)...... (0,mfcc)]
    for i in tqdm(WORDS, desc="Extract_MFCC"):
        path =list(pathlib.Path('C:/Users/김남형/Desktop/연구/연구소스/data/speech_commands_v0.01/'+i+'/').glob('*.wav'))
        mfccs_set = []
        for j  in path:
            mfccs_set.append((0,mfcc_from_wavpath(j.as_posix())))
        mfccs_dict[i] = copy.deepcopy(mfccs_set)

    #각 word의 initial 센트로이드 설정
    for i in tqdm(WORDS, desc = "initialize_centroid"):
        mfccs_set = []
        for j in range(codebook_size_of_each_word):
            word_mfccs_set = copy.deepcopy(mfccs_dict[i])
            mfccs_set.append((j,copy.deepcopy(word_mfccs_set[j][1])))
        mfccs_cluster_dict[i] = copy.deepcopy(mfccs_set)

    # mfccs_cluster_dict : initial 센트로이드들 들어있음.
    for loop in tqdm(range(itera)):     
    #k means 시작 지금은 루프를 씌워서 작동함.
        for key in WORDS:
        # Word 별로 kmeans 하므로 가져온다.
            key_mfccs_dict = copy.deepcopy(mfccs_dict[key]) #type = tuples array
            key_mfccs_cluster_dict = copy.deepcopy(mfccs_cluster_dict[key])#type =tuples array
            processed_key_mfccs_dict = list()

            for mth_key_mfccs_dict in key_mfccs_dict:
                tmp=0 # k번 반복한 후에 다시 0으로 바뀌기 때문에 이거 하나면 된다
                for kth_key_mfccs_cluster_dict in key_mfccs_cluster_dict:
                
                    path_of_kmth, distance_of_kmth =dtw_path(kth_key_mfccs_cluster_dict[1],mth_key_mfccs_dict[1])
                    if tmp == 0:
                        selected_path_of_mth = copy.deepcopy(path_of_kmth)
                        selected_distance_of_mth = distance_of_kmth
                        selected_cluster_of_mth = kth_key_mfccs_cluster_dict[0]
                        tmp = 1
                        continue 
                    if distance_of_kmth <= selected_distance_of_mth:
                        selected_path_of_mth = copy.deepcopy(path_of_kmth)
                        selected_distance_of_mth = distance_of_kmth
                        selected_cluster_of_mth = kth_key_mfccs_cluster_dict[0]
            
                processed_mfccs= make_ylength_to_xlength_with_dtw(key_mfccs_cluster_dict[selected_cluster_of_mth][1],mth_key_mfccs_dict[1],selected_path_of_mth)
                processed_key_mfccs_dict.append((selected_cluster_of_mth,copy.deepcopy(processed_mfccs)))
        
            number_of_clusters_sample = [1 for i in range(codebook_size_of_each_word)]

            #각 클러스터의 갯수를 세야된다. 그다음에 key_mfccs_cluster_dict에 더해줘야된다.
            for i in processed_key_mfccs_dict:
                number_of_clusters_sample[i[0]] += 1
                key_mfccs_cluster_dict[i[0]] = copy.deepcopy((i[0],copy.deepcopy(np.add(i[1],key_mfccs_cluster_dict[i[0]][1]))))
        
            #이제 클러스터 갯수만큼 나눠준다
            for alpha in key_mfccs_cluster_dict:
                key_mfccs_cluster_dict[alpha[0]]= copy.deepcopy((alpha[0],alpha[1]/number_of_clusters_sample[alpha[0]]))
        
            #원본에다가 저장해야된다.
            mfccs_cluster_dict[key]=copy.deepcopy(key_mfccs_cluster_dict)
    #변환
    for key in WORDS:
        dumy = [CODEBOOK.append((key,i[1])) for i in mfccs_cluster_dict[key]]
        #저장을 안할거지만 list comprehension 때문에 생성한 변수임 dummy는.
    

                
                
                
    
def make_ylength_to_xlength_with_dtw(x,y,path):
    '''
    x,y => mfcc_timeseries, path => dtw_path's path
    '''
    p= []
    q= []
    for i in range(len(path)):
        a=path[i]
        p.append(a[0])
        q.append(a[1])
    p = np.array(p)
    q = np.array(q)
    path = (p,q)

    yp = np.zeros(x.shape)
    for i in range(len(path[0])):
        ix, iy = path[0][i], path[1][i]
        yp[ix] = copy.deepcopy(y[iy])
    
    return yp



'''###########################
EVALUATE
###########################'''
def evaulate_normal_dtw(codebook_size_of_each_word=5, order ='fix'):
    
    set_codebook_normal(codebook_size_of_each_word=codebook_size_of_each_word,order=order)
    path = './data/speech_commands_v0.01/testing_list.txt'
    
    speech_tuple = load_filepath_for_speech_commands('./data/speech_commands_v0.01/testing_list.txt','')
    fail =0
    success =0
    
    for i in tqdm(speech_tuple, desc="verifying performance of normal-dtw"):
        yhat = guess_tuple(i[1])
        y = i[0]
        if yhat == y:
            success += 1
        else:
            fail += 1
            #print(y,yhat)
    print("success: ",success)
    print("fail:",fail)

def evaulate_k_dtw(codebook_size_of_each_word=5, k_means_iter =10):

    set_codebook_with_dtw_k_means(codebook_size_of_each_word=codebook_size_of_each_word,itera = k_means_iter)
   
    speech_tuple = load_filepath_for_speech_commands('./data/speech_commands_v0.01/testing_list.txt','')
    fail =0
    success =0

    for i in tqdm(speech_tuple, desc='verifying performance of k-means clustering. k='+str(codebook_size_of_each_word)):
        yhat = guess_tuple(i[1])
        y = i[0]
        if yhat == y:
            success += 1
        else:
            fail += 1
            #print(y,yhat)
    print("success: ",success)
    print("fail:",fail)


if __name__ == "__main__":
    #set_codebook_with_dtw_k_means(3,10)
    evaulate_normal_dtw(codebook_size_of_each_word=10, order='fix')
    #evaulate_k_dtw(codebook_size_of_each_word=10, k_means_iter=50)
