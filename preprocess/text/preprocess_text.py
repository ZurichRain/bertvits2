import os
# import copyfile
from shutil import copyfile

from cleaner import clean_text
# '/data/hypertext/sharpwang/dataset/leidian'
# with open()
def clean_file(ori,dst,save_file,name):
    rowid = 0
    clean_ans = []

    for file in os.listdir(ori):
        # print(file)
        if file == name+'id':
            continue
        text = file.split('.')[0].split('_')[-1].strip()
        # print(text)
        copyfile(ori+file, dst+ name+'_'+str(rowid)+'.wav')

        norm_text, phones, tones, word2ph = clean_text(text,'ZH')

        phones = [str(i) for i in phones]
        tones = [str(i) for i in tones]
        word2ph = [str(i) for i in word2ph]

        clean_ans.append('|'.join([name+'_'+str(rowid),'ZH', text,' '.join(phones), ' '.join(tones), ' '.join(word2ph)]))

        rowid+=1

        # print(norm_text, phones, tones, word2ph)

    with open(save_file+ name+'.cleaned','w',encoding='utf-8') as f:
        for ans in clean_ans:
            f.write(ans+'\n')

save_file = '/data/hypertext/sharpwang/dataset/TTS/leidain/'
name='leidian'
ori = '/data/hypertext/sharpwang/dataset/leidian/'
dst = '/data/hypertext/sharpwang/dataset/TTS/leidain/audioid_wav_dir/'

clean_file(ori,dst,save_file,name)