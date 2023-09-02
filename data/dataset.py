import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import utils.commons as commons
from utils.mel_processing import spectrogram_torch, mel_spectrogram_torch, spec_to_mel_torch
from utils.utils import load_wav_to_torch, load_filepaths_and_text
from preprocess.text import cleaned_text_to_sequence

"""Multi speaker version"""

'''
datadir:
    audioid_wav_dir:
    text_bert_dir:
    audio_spec_dir:
    list_audio_text_pair_file
'''


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, 
                args,
                rank):
        self.datatype = args.datatype
        self.datadir = args.datadir
        self.rank=rank

        if self.datatype=='train':
            audio_text_pair_file = self.datadir+'train_list_audio_text_pair_file.txt'
        elif self.datatype=='val':
            audio_text_pair_file = self.datadir+'val_list_audio_text_pair_file.txt'
        elif self.datatype=='all':
            audio_text_pair_file = self.datadir+'list_audio_text_pair_file.txt'

        self.audioTextData = load_filepaths_and_text(audio_text_pair_file)
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.filter_length = args.filter_length
        self.hop_length = args.hop_length
        self.win_length = args.win_length
        self.sampling_rate = args.sampling_rate
        self.spk_map = args.spk2id
        self.args = args

        self.use_mel_spec_posterior = getattr(args, "use_mel_posterior_encoder", False)
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(args, "n_mel_channels", 80)

        self.cleaned_text = getattr(args, "cleaned_text", False)

        self.add_blank = args.add_blank
        self.min_text_len = getattr(args, "min_text_len", 1)
        self.max_text_len = getattr(args, "max_text_len", 300)

        # random.seed(1234)
        # random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def get_audio_path(self, audio_id):
        # dirname = os.path.join('./esd/ESD/', spk.split('_')[0])
        # dirname = os.path.join(dirname, spk.split('_')[1])

        wav_file = os.path.join(self.datadir+'audioid_wav_dir', audio_id+'.wav')

        return wav_file

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        audioTextData_new = []
        lengths = [] 
        skipped = 0
        for audio_id, language, text, phones, tone, word2ph in self.audioTextData:
            
            if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
                phones = phones.split(" ")
                tone = [int(i) for i in tone.split(" ")]
                word2ph = [int(i) for i in word2ph.split(" ")]
                # ori_audiopath = audiopath
                
                # try:
                #     self.get_audio(audiopath)
                # except:
                #     skipped += 1
                #     continue
                audioTextData_new.append([audio_id, language, text, phones, tone, word2ph])

                # print(audiopaths_sid_text_new[-1])
                audiopath = self.get_audio_path(audio_id)
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else:
                skipped += 1
        if self.rank==0:
            print("skipped: ", skipped, ", total: ", len(self.audioTextData))

        self.audioTextData = audioTextData_new
        self.lengths = lengths # 数据中的audio分布 之后如果按照audio的长度来建立bath的时候需要用到

    def get_audio_text_speaker_pair(self, audioTextData):
        # separate filename, speaker_id and text
        audioid, language, text, phones, tone, word2ph = audioTextData
        # print(audiopath_sid_text)
        # audiopath = self.get_audio_path(audioid)
        text_bert_dir=os.path.join(self.datadir+"text_bert_dir/", audioid+'.bert.pt')
        text_bert, phones, tone, language = self.get_text(text_bert_dir, language, text, phones, tone, word2ph)

        # print(audiopath, sid, language, text, phones, tone, word2ph)
        audiopath = self.get_audio_path(audioid)
        spec_filename=os.path.join(self.datadir+"audio_spec_dir/", audioid+'.spec.pt')
        spec, wav = self.get_audio(audiopath,spec_filename)
        # sid = torch.LongTensor([int(self.spk_map[sid])])
        sid = torch.LongTensor([172])
        # return (phones, spec, wav, sid, tone, language, text_bert)
        return {
            'text_bert':text_bert,
            'phones':phones,
            'tone':tone,
            'language':language,
            'word2ph':word2ph,
            'spec':spec,
            'wav':wav,
            'sid':sid
        }

    def get_audio(self, filename, spec_filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        # spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            if self.use_mel_spec_posterior:
                # if os.path.exists(filename.replace(".wav", ".spec.pt")):
                #     # spec, n_fft, num_mels, sampling_rate, fmin, fmax
                #     spec = spec_to_mel_torch(
                #         torch.load(filename.replace(".wav", ".spec.pt")), 
                #         self.filter_length, self.n_mel_channels, self.sampling_rate,
                #         self.hparams.mel_fmin, self.hparams.mel_fmax)
                spec = mel_spectrogram_torch(audio_norm, self.filter_length,
                    self.n_mel_channels, self.sampling_rate, self.hop_length,
                    self.win_length, self.hparams.mel_fmin, self.hparams.mel_fmax, center=False)
            else:
                spec = spectrogram_torch(audio_norm, self.filter_length,
                    self.sampling_rate, self.hop_length, self.win_length,
                    center=False)
            spec = torch.squeeze(spec, 0)
            if self.rank == 0:
                torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text_bert_dir, language, text, phones, tone, word2ph):
        
        # print(text, word2ph,phone, tone, language_str)

        # pold = phone
        # w2pho = [i for i in word2ph]
        # word2ph = [i for i in word2ph]
        phone, tone, language = cleaned_text_to_sequence(phones, tone, language)
        # 这里得到了 不同的编码
        # pold2 = phone

        if self.add_blank:
            # 添加空的语调
            # p1 = len(phone)
            phone = commons.intersperse(phone, 0)
            # p2 = len(phone)
            # t1 = len(tone)
            tone = commons.intersperse(tone, 0)
            # t2 = len(tone)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        # print(wav_path)
        # bert_path = audiopath.replace(".wav", ".bert.pt")
        # print(bert_path)
        try:
            text_bert = torch.load(text_bert_dir)
            assert text_bert.shape[-1] == len(phone)
        except:
            #bert = get_bert(text, word2ph, language_str)
            #torch.save(bert, bert_path)
            # print(bert.shape[-1], text_bert_dir, text, pold)
            exit()
            #assert bert.shape[-1] == len(phone)

        # assert bert.shape[-1] == len(phone), (
        # bert.shape, len(phone), sum(word2ph), p1, p2, t1, t2, pold, pold2, word2ph, text, w2pho)
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return text_bert, phone, tone, language

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audioTextData[index])

    def __len__(self):
        return len(self.audioTextData)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # (phones, spec, wav, sid, tone, language, text_bert)
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x['spec'].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x['phones']) for x in batch])
        max_spec_len = max([x['spec'].size(1) for x in batch])
        max_wav_len = max([x['wav'].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)

        spec_padded = torch.FloatTensor(len(batch), batch[0]['spec'].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        bert_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row['phones']
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row['spec']
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row['wav']
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row['sid']

            tone = row['tone']
            tone_padded[i, :tone.size(0)] = tone

            language = row['language']
            language_padded[i, :language.size(0)] = language

            bert = row['text_bert']
            bert_padded[i, :, :bert.size(1)] = bert

        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, tone_padded, language_padded, bert_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]

        # self.lengths 中存储的是每个音频的size
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
