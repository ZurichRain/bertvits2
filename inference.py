import sys
sys.path.append('.')
from io import BytesIO

# import ffmpeg
import base64
import os
import sys
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import time
from utils import commons
from utils import utils
from model.models import SynthesizerTrn
from preprocess.text.symbols import symbols
from preprocess.text import cleaned_text_to_sequence,_symbol_to_id, get_bert
from preprocess.text.cleaner import clean_text
from scipy.io import wavfile

# Get ffmpeg path
# ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg")

def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    print([f"{p}{t}" for p, t in zip(phone, tone)])
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str)

    assert bert.shape[-1] == len(phone)

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    return bert, phone, tone, language

def infer(text, sdp_ratio, noise_scale, noise_scale_w,length_scale,sid):
    bert, phones, tones, lang_ids = get_text(text,"ZH", hps,)
    with torch.no_grad():
        x_tst=phones.to(dev).unsqueeze(0)
        tones=tones.to(dev).unsqueeze(0)
        lang_ids=lang_ids.to(dev).unsqueeze(0)
        bert = bert.to(dev).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(dev)
        speakers = torch.LongTensor([172]).to(dev)
        audio = net_g.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids,bert, sdp_ratio=sdp_ratio
                           , noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
        return audio

def replace_punctuation(text, i=2):
    punctuation = "，。？！"
    for char in punctuation:
        text = text.replace(char, char * i)
    return text

# Load Generator
hps = utils.get_hparams_from_file("./configs/config.json")

dev='cuda'
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(dev)
_ = net_g.eval()
ckp_path="/data/hypertext/sharpwang/TTS/Bert-VITS2/logs/wfleidain/G_2000.pth"
# ckp_path = '/data/hypertext/sharpwang/LLM/BertVits2/G_0.pth'
_ = utils.load_checkpoint(ckp_path, net_g, None,skip_optimizer=True)


speaker = 0
text = "刘奶奶找牛奶奶买牛奶，牛奶奶给刘奶奶拿牛奶，"
sdp_ratio = 0.2
noise = 0.5
noisew = 0.6
length = 1.2

with torch.no_grad():
    audio = infer(text, sdp_ratio=sdp_ratio, noise_scale=noise, noise_scale_w=noisew, length_scale=length, sid=speaker)

# wav = BytesIO()
wavfile.write('test.wav', hps.data.sampling_rate, audio)
torch.cuda.empty_cache()