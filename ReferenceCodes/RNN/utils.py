import os
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
import random
from gensim.models import Word2Vec
import pickle
import openpyxl
import re
from copy import deepcopy
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm


ChKrEntity = namedtuple('entity', 'chinese ch_idx type korean kr_idx')

kings_total = ['k01_Taejo', 'k02_Jeongjong', 'k03_Taejong', 'k04_Sejong', 'k05_Munjong', 'k06_Danjong',
               'k07_Sejo', 'k08_Yejong', 'k09_Seongjong', 'k10_Yeonsangun', 'k11_Jungjong', 'k12_Injong',
               'k13_Myungjong', 'k14_Seonjo', 'k14_Seonjo_sujeong', 'k15_Gwanghaegun_jeongcho',
               'k15_Gwanghaegun_jungcho', 'k16_Injo', 'k17_Hyojong', 'k18_Hyeonjong', 'k18_Hyeonjong_gaesu',
               'k19_Sukjong', 'k19_Sukjong_bogwoljeongo', 'k20_Gyeongjong', 'k20_Gyeongjong_sujeong',
               'k21_Yeongjo', 'k22_Jeongjo', 'k23_Sunjo', 'k24_Heonjong', 'k25_Cheoljong']

HOME_PATH = os.path.expanduser('~')
SILOK_PATH = os.path.join(HOME_PATH, 'chosun-utf8')

PAD = 'PAD'
UNK = 'UNK'
GO = 'GO'
EOS = 'EOS'
EOS_ID = 3
# SPECIAL_WORDS = [PAD, UNK, GO, EOS]

CONFIG_TXT = 'config.txt'
CONFIG_PICKLE = 'config.p'
LOG_DIR = 'Logs'
PARAMS_DIR = 'Params'
PARAMS = 'params'
TRAIN_LOG = 'Train_log.txt'


not_ch = re.compile(r'[^\u2E80-\u2EFF\u3400-\u4DBF\u4E00-\u9FBF\uF900-\uFAFF]')
space = re.compile(r'\s{2,}')

# ###########################################################################
# ################## For NER ################################################
# ###########################################################################
T_NAME = '이름'
T_BOOK = '서명'
T_YEAR = '연호'
T_LOC = '지명'

TYPES = [T_NAME, T_LOC, T_BOOK, T_YEAR]

NAME_REGEX = re.compile(r'12*')
LOC_REGEX = re.compile(r'34*')
BOOK_REGEX = re.compile(r'56*')
YEAR_REGEX = re.compile(r'78*')


TYPES_REGEX = [NAME_REGEX, LOC_REGEX, BOOK_REGEX, YEAR_REGEX]

OUTSIDE = 'OUTSIDE'
B_NAME = 'B_NAME'
I_NAME = 'I_NAME'
B_LOC = 'B_LOC'
I_LOC = 'I_LOC'
B_BOOK = 'B_BOOK'
I_BOOK = 'I_BOOK'
B_YEAR = 'B_YEAR'
I_YEAR = 'I_YEAR'
BIO2IDX = {OUTSIDE: 0,
           B_NAME: 1, I_NAME: 2,
           B_LOC: 3, I_LOC: 4,
           B_BOOK: 5, I_BOOK: 6,
           B_YEAR: 7, I_YEAR: 8}

TYPE2IDX = {T_NAME: (BIO2IDX[B_NAME], BIO2IDX[I_NAME]),
            T_LOC: (BIO2IDX[B_LOC], BIO2IDX[I_LOC]),
            T_BOOK: (BIO2IDX[B_BOOK], BIO2IDX[I_BOOK]),
            T_YEAR: (BIO2IDX[B_YEAR], BIO2IDX[I_YEAR])}


CODE2KING = {'waa': (1, 'k01_Taejo'),
             'wba': (2, 'k02_Jeongjong'),
             'wca': (3, 'k03_Taejong'),
             'wda': (4, 'k04_Sejong'),
             'wea': (5, 'k05_Munjong'),
             'wfa': (6, 'k06_Danjong'),
             'wga': (7, 'k07_Sejo'),
             'wha': (8, 'k08_Yejong'),
             'wia': (9, 'k09_Seongjong'),
             'wja': (10, 'k10_Yeonsangun'),
             'wka': (11, 'k11_Jungjong'),
             'wla': (12, 'k12_Injong'),
             'wma': (13, 'k13_Myungjong'),
             'wna': (14, 'k14_Seonjo'),
             'wnb': (14, 'k14_Seonjo_sujeong'),
             'woa': (15, 'k15_Gwanghaegun_jungcho'),
             'wob': (15, 'k15_Gwanghaegun_jeongcho'),
             'wpa': (16, 'k16_Injo'),
             'wqa': (17, 'k17_Hyojong'),
             'wra': (18, 'k18_Hyeonjong'),
             'wrb': (18, 'k18_Hyeonjong_gaesu'),
             'wsa': (19, 'k19_Sukjong'),
             'wsb': (19, 'k19_Sukjong_bogwoljeongo'),
             'wta': (20, 'k20_Gyeongjong'),
             'wtb': (20, 'k20_Gyeongjong_sujeong'),
             'wua': (21, 'k21_Yeongjo'),
             'wva': (22, 'k22_Jeongjo'),
             'wwa': (23, 'k23_Sunjo'),
             'wxa': (24, 'k24_Heonjong'),
             'wya': (25, 'k25_Cheoljong')}
# ###########################################################################
# ################## For NER ################################################
# ###########################################################################


class PairData:
    def __init__(self,
                 ch_sent: str,
                 ch_tok: List[str],
                 kr_sent: str,
                 kr_tok: List[str]):
        self.ch_sent = ch_sent
        self.ch_tok = ch_tok
        self.ch_seq_len = len(self.ch_tok)
        self.kr_sent = kr_sent
        self.kr_tok = kr_tok
        self.kr_seq_len = len(self.kr_tok)

    def print(self):
        s = """
            Chinese
                sentence: %s
                token: %s
                sequence length: %s
            Korean
                sentence: %s
                token: %s
                sequence length: %s
            \n\r""" % (self.ch_sent, self.ch_tok, self.ch_seq_len,
                       self.kr_sent, self.kr_tok, self.kr_seq_len)
        return s


class DataSet:
    """
    Encoder Input: Sequence of Chinese Character Tokens
    Decoder Input: Sequence of Korean Tokens
    """
    def __init__(self,
                 enc_max_len: int,
                 dec_max_len: int,
                 ch_word2id: Dict,
                 kr_word2id: Dict):
        self.enc_max_len = enc_max_len
        # include GO or EOS token
        self.dec_max_len = dec_max_len

        self.ch_word2id = ch_word2id
        self.kr_word2id = kr_word2id

        self._sequential_indices = 0

        self.num_data = None
        self.full_indices = None
        self.enc_tokens = []
        self.enc_inputs = None
        self.enc_seq_len = None

        self.dec_tokens = []
        self.dec_inputs = None
        self.dec_targets = None
        self.dec_seq_len = None

    def make(self,
             pairs: List[PairData],
             train_phase: bool):

        new_pairs = list()
        for p in pairs:
            # GO or EOS token will be appended to Korean sequence
            if 1 <= p.ch_seq_len <= self.enc_max_len and 1 <= p.kr_seq_len <= self.dec_max_len - 1:
                new_pairs.append(p)
        del pairs

        self.num_data = len(new_pairs)
        self.full_indices = list(range(self.num_data))
        self.enc_inputs = np.zeros([self.num_data, self.enc_max_len], dtype=np.uint32)
        self.enc_seq_len = np.zeros([self.num_data], dtype=np.uint32)

        self.dec_inputs = np.zeros([self.num_data, self.dec_max_len], dtype=np.uint32)
        self.dec_targets = np.zeros([self.num_data, self.dec_max_len], dtype=np.uint32)
        self.dec_seq_len = np.zeros([self.num_data], dtype=np.uint32)

        for i, p in enumerate(new_pairs):
            # ch_ids = list(map(lambda x: self.ch_word2id.get(x, self.ch_word2id[UNK]), p.ch_tok))
            ch_ids = seq2ids(p.ch_tok, self.ch_word2id)
            self.enc_inputs[i, :p.ch_seq_len] = ch_ids
            self.enc_seq_len[i] = p.ch_seq_len

            # kr_ids = list(map(lambda x: self.kr_word2id.get(x, self.kr_word2id[UNK]), p.kr_tok))
            kr_ids = seq2ids(p.kr_tok, self.kr_word2id)
            self.dec_inputs[i, :p.kr_seq_len+1] = [self.kr_word2id[GO]] + kr_ids
            self.dec_targets[i, :p.kr_seq_len+1] = kr_ids + [self.kr_word2id[EOS]]
            self.dec_seq_len[i] = p.kr_seq_len + 1

            if not train_phase:
                self.enc_tokens.append(p.ch_tok)
                self.dec_tokens.append(p.kr_tok)

    def get_train_batch(self, batch_size):
        indices = random.sample(self.full_indices, batch_size)
        out = (self.enc_inputs[indices, :],
               self.enc_seq_len[indices],
               self.dec_inputs[indices, :],
               self.dec_targets[indices, :],
               self.dec_seq_len[indices])
        return out

    @property
    def sequential_indices(self):
        return self._sequential_indices

    @sequential_indices.setter
    def sequential_indices(self, idx=0):
        self._sequential_indices = idx

    def get_eval_batch(self, batch_size):
        s = self.sequential_indices
        e = self.sequential_indices + batch_size
        # If remain data is not enough, < batch size, drop remain data, finish sequential batch
        out = (self.enc_tokens[s:e],
               self.enc_inputs[s:e, :],
               self.enc_seq_len[s:e],
               self.dec_tokens[s:e])
        self.sequential_indices += batch_size
        return out

    def get_check_batch(self, batch_size):
        s = self.sequential_indices
        e = self.sequential_indices + batch_size
        out = (self.enc_tokens[s:e],
               self.enc_inputs[s:e, :],
               self.enc_seq_len[s:e],
               self.dec_tokens[s:e],
               self.dec_inputs[s:e, :],
               self.dec_targets[s:e, :],
               self.dec_seq_len[s:e],)
        self.sequential_indices += batch_size
        return out


class ParaData:
    """
    XML file paragraph data
    """
    def __init__(self,
                 chinese: str,
                 level5_id: str,
                 y: np.ndarray,
                 seq_len: int,
                 entities: List[Tuple[str, int, str]]):
        self.chinese = chinese
        self.level5_id = level5_id
        self.y = y
        self.seq_len = seq_len
        self.entities = entities

    def print(self):
        s = """
            Chinese:    %s
            level5_id:  %s
            Y:          %s
            seq_len:    %s
            entities:   %s
            restore-e:  %s
            \r""" % (self.chinese,
                     self.level5_id,
                     list(self.y),
                     self.seq_len,
                     self.entities,
                     pred2ne(self.chinese, self.y))
        return s


class DataSetNER:
    """
    Input: Chinese Character
    Output: Begin, Inside, Outside
    """
    def __init__(self,
                 max_seq_len: int,
                 char2id: Dict):

        self.max_seq_len = max_seq_len
        self.char2id = char2id

        self._sequential_indices = 0

        self.num_data = None
        self.full_indices = None

        self.ch_toks = []
        self.x = None
        self.y = None
        self.seq_len = None

    def make(self,
             paras: List[ParaData],
             train_phase: bool):

        new_paras = []
        for p in paras:
            # GO or EOS token will be appended to Korean sequence
            if 1 <= p.seq_len <= self.max_seq_len:
                new_paras.append(p)
        del paras

        self.num_data = len(new_paras)
        self.full_indices = list(range(self.num_data))
        self.x = np.zeros([self.num_data, self.max_seq_len], dtype=np.uint16)
        self.y = np.zeros([self.num_data, self.max_seq_len], dtype=np.uint16)
        self.seq_len = np.zeros([self.num_data], dtype=np.uint16)

        for i, p in enumerate(new_paras):
            ch_ids = seq2ids(list(p.chinese), self.char2id)

            if not (len(ch_ids) == len(p.y) == p.seq_len):
                raise ValueError('Sequence Length Not Equal x: %d,  y: %d,  seq_len: %d' %
                                 (len(ch_ids), len(p.y), p.seq_len))

            self.x[i, :p.seq_len] = ch_ids
            self.y[i, :p.seq_len] = p.y
            self.seq_len[i] = p.seq_len

            if not train_phase:
                self.ch_toks.append(list(p.chinese))
            print('\rMake Dataset... %d/%d' % (i, self.num_data), end='')
        print()
        return

    def get_train_batch(self, batch_size):
        """
        :param batch_size: batch size
        :return: x, y, seq_len
        """
        indices = random.sample(self.full_indices, batch_size)
        x = self.x[indices, :]
        y = self.y[indices, :]
        seq_len = self.seq_len[indices]
        return x, y, seq_len

    @property
    def sequential_indices(self):
        return self._sequential_indices

    @sequential_indices.setter
    def sequential_indices(self, idx=0):
        self._sequential_indices = idx

    def get_eval_batch(self, batch_size):
        s = self.sequential_indices
        e = self.sequential_indices + batch_size
        # If remain data is not enough, < batch size, drop remain data, finish sequential batch
        ch_toks = self.ch_toks[s:e]
        x = self.x[s:e, :]
        y = self.y[s:e, :]
        seq_len = self.seq_len[s:e]
        self.sequential_indices += batch_size
        return ch_toks, x, y, seq_len


def clean_up_ch(ch: str):
    """
    Remove all character which is not Chinese.
    """
    return not_ch.sub('', ch)


def pickle_store(obj, path):
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()
    return


def pickle_load(path):
    f = open(path, 'rb')
    a = pickle.load(f)
    f.close()
    return a


def read_xlsx(xlsx_path, column: List):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    data = []
    if len(column) == 1:
        for r in ws.rows:
            data.append(r[column[0]].value)
    else:
        for r in ws.rows:
            rr = []
            for c in column:
                rr.append(r[c].value)
            data.append(tuple(rr))
    wb.close()
    return data


def read_pickles(folder: str):
    files = os.listdir(folder)
    d = []
    for f in files:
        if f.split('.')[-1] == 'p':
            d = d + pickle_load(os.path.join(folder, f))
    return d


def print_write(s, file, mode=None):
    if isinstance(file, str):
        if mode is None:
            mode = 'a'
        f = open(file, mode)
        print(s, end='')
        f.write(s)
        f.close()
    else:
        print(s, end='')
        file.write(s)


def dict_swap(my_dic: Dict):
    return {v: k for k, v in my_dic.items()}


def uni2dec(c):
    encoded = list(c.encode('unicode-escape'))[2:]
    if not encoded:
        return 0
    h = ''
    for c in encoded:
        h += chr(c)
    return int(h, 16)


def unicode_classify(c):
    dec = uni2dec(c)
    if (int('4E00', 16) <= dec <= int('9FFF', 16) or
            int('3400', 16) <= dec <= int('4DBF', 16) or
            int('F900', 16) <= dec <= int('FAFF', 16)):
        return 0    # Chinese
    elif int('AC00', 16) <= dec <= int('D7AF', 16):
        return 1    # Korean
    else:
        return 2    # else


def text_readlines(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    t = f.readlines()
    f.close()
    return t


def text_read(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    t = f.read()
    f.close()
    return t


def read_dic(path, start_line):
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()[start_line:]
    f.close()
    dic = dict()
    for i, l in enumerate(lines):
        try:
            s = l.split()
            dic[s[0]] = s[1]
        except IndexError:
            print(i)
            print(lines[i])
            print(lines[i - 1])
            raise IndexError
    return dic


def read_lexicon(path, start_line):
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()[start_line:]
    f.close()
    o = list()
    for i, l in enumerate(lines):
        try:
            s = l.split()
            o.append((s[0], s[1]))
        except IndexError:
            print(i)
            print(lines[i])
            print(lines[i-1])
            raise IndexError
    return o


def chinese_regularize(line, dic):
    """
    Change all chinese character unified
    :param line: Chinese or Korean that has chinese character sentence
    :param dic: compatibility -> unified chinese character dictionary
    :return: unified chinese sentence
    """
    # line_out = ''
    # for c in line:
    #     try:
    #         line_out = line_out + dic[c]
    #     except KeyError:
    #         line_out = line_out + c
    # return line_out
    return ''.join(list(map(lambda x: dic.get(x, x), line)))


def remove_redundant_line(lines: List[str], check=0):
    new_lines = []
    for line in lines:
        f = False
        for c in line:
            if unicode_classify(c) == check:    # 0 hanja, 1 hangeul
                f = True
                break
        if f:
            new_lines.append(line)
    return new_lines


def remove_single_entity(ne_list):
    new = []
    for n in ne_list:
        if isinstance(n, tuple):
            if len(n[0]) > 1:
                new.append(n)
        else:
            if len(n) > 1:
                new.append(n)
    return new


def get_txt_paths(basic_path: str, kings: List[str]):
    txts = []
    for king in kings:
        king_path = os.path.join(basic_path, king)
        books = os.listdir(king_path)
        for book in books:
            book_path = os.path.join(king_path, book)
            texts = os.listdir(book_path)
            for text in texts:
                text_path = os.path.join(book_path, text)
                txts.append(text_path)
    return txts


def seq2ids(seq: List[str], d: Dict) -> List[int]:
    ids = list(map(lambda x: d.get(x, d[UNK]), seq))
    return ids


def ids2seq(ids: Union[np.ndarray, List[int]], d: Dict, clip_token: Optional[str] = None):
    seq = list(map(lambda x: d[x], ids))
    if clip_token is not None:
        try:
            return seq[:seq.index(clip_token)]
        except ValueError:
            return seq
    else:
        return seq


def load_word2vec(model: Union[Word2Vec, str],
                  special_words: List[str]) -> Tuple[Dict, np.ndarray, int]:
    """
    :param model: gensim word2vec model
    :param special_words: List of special word such as Pad, Unknown, Go and EOS token
    :return: word dictionary, embedding vectors, embedding size
    """
    if isinstance(model, str):
        model = Word2Vec.load(model)

    size = model.wv.vectors.shape[1]

    special_vectors = list()
    m = np.mean(model.wv.vectors)
    s = np.std(model.wv.vectors)
    for word in special_words:
        if word == PAD:
            special_vectors.append(np.zeros((1, size)))
        else:
            special_vectors.append(np.random.normal(size=(1, size), loc=m, scale=s))

    special_vectors = np.concatenate(special_vectors, axis=0)
    assert special_vectors.shape[0] == len(special_words)

    word2idx = {c: i for i, c in enumerate(special_words + model.wv.index2word)}
    vectors = np.concatenate((special_vectors, model.wv.vectors), axis=0)

    return word2idx, vectors, size


def pred2ne(sent: str, pred: Union[np.ndarray, List, Tuple]):
    if len(sent) != len(pred):
        raise ValueError('sentence length(%d) and prediction length(%d) is not equal'
                         % (len(sent), len(pred)))

    seq_len = sent.find(PAD)
    if seq_len < 0:
        seq_len = len(sent)
    sent = sent[:seq_len]
    pred = pred[:seq_len]
    pred_seq = ''.join(map(lambda x: str(x), pred))

    entities = []

    for r, t in zip(TYPES_REGEX, TYPES):
        for m in r.finditer(pred_seq):
            start, end = m.span()
            entities.append((sent[start:end], start, t))

    entities.sort(key=lambda x: x[1])
    return entities


def entity_form(target: List[List[Tuple[str, int, str]]],
                prediction: List[List[Tuple[str, int, str]]]) -> Tuple[float, float, float]:
    assert len(target) == len(prediction)
    n_sentence = len(target)
    true_ne = deepcopy(target)
    pred_ne = deepcopy(prediction)

    tp_ne = []
    fp_ne = []
    fn_ne = []

    for i in range(n_sentence):
        p = set(pred_ne[i])
        t = set(true_ne[i])

        tp_ne.append(p & t)
        fp_ne.append(p - t)
        fn_ne.append(t - p)

    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(n_sentence):
        true_positive += len(tp_ne[i])
        false_positive += len(fp_ne[i])
        false_negative += len(fn_ne[i])

    return score(true_positive, false_positive, false_negative)


def surface_form(target: List[List[Tuple[str, int, str]]],
                 prediction: List[List[Tuple[str, int, str]]]) -> Tuple[float, float, float]:
    assert len(target) == len(prediction)
    n_sentence = len(target)
    true_ne = []
    pred_ne = []

    for i in range(n_sentence):
        for ne in target[i]:
            true_ne.append(ne[0])
        for ne in prediction[i]:
            pred_ne.append(ne[0])

    t = set(true_ne)
    p = set(pred_ne)

    tp_ne = p & t
    fp_ne = p - t
    fn_ne = t - p

    true_positive = len(tp_ne)
    false_positive = len(fp_ne)
    false_negative = len(fn_ne)

    return score(true_positive, false_positive, false_negative)


def harmonic_mean(a, b):
    return (2*a*b)/(a+b)


def score(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0

    try:
        f1 = harmonic_mean(precision, recall)
    except ZeroDivisionError:
        f1 = 0

    return f1, precision, recall


def search_entity_by_dictionary(line, ne_list):
    # ne_list = [('綾城監', '이름'), ('尹熙鼎', '이름'), ('文尙男', '이름'), ('聖賚', '이름')]
    entities = []
    temp = line
    for ne, c in ne_list:
        flag = False
        s = 0
        while True:
            ne_idx = temp.find(ne, s)
            if ne_idx < 0:
                break
            else:
                flag = True
                s = ne_idx + len(ne)
                entities.append((ne, ne_idx, c))
        if flag:
            temp = temp.replace(ne, '-'*(len(ne)))
    return entities


def entity_lookup(attention_map, chinese_entities, kr_tokens):
    # attention_map [decoder length, encoder length]
    # entities = [('徐便己', 36, '이름'), ('胡璟', 43, '이름'),
    # ('龍州', 52, '지명'), ('李勇', 66, '이름'), ('《綱目》', 43, '서명')]
    entities = []
    for entity_tuple in chinese_entities:
        entity = entity_tuple[0]
        ch_idx = entity_tuple[1]
        entity_type = entity_tuple[2]

        if entity_type == '서명':
            entity = entity.strip('《》')
            ch_idx = ch_idx+1

        a = attention_map[:, ch_idx:ch_idx+len(entity)]
        s = np.sum(a, axis=1)
        e = ChKrEntity(chinese=entity,
                       ch_idx=ch_idx,
                       type=entity_type,
                       korean=kr_tokens[np.argmax(s)],
                       kr_idx=np.argmax(s))
        entities.append(e)
        # entities.append((kr_tokens[np.argmax(s)], np.argmax(s)))
    return entities


def entity_replace(hypothesis: List[str], entities: List[ChKrEntity], entity_dic: Dict):
    hyp_new = hypothesis[:]
    for entity in entities:
        try:
            kr = entity_dic[entity.chinese]
            hyp_new[entity.kr_idx] = kr
        except KeyError:
            pass
    return hyp_new


def matplotlib_fonts():
    font_list = [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]

    # path = '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'

    # font_name = fm.FontProperties(fname=path).get_name()
    print(font_list)
    # print(font_name)

    print([(f.name, f.fname) for f in fm.fontManager.ttflist])

    print(fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'))


def plot_attention(attention_map, input_tags=None, output_tags=None, title='attention_map'):
    """
    :param attention_map: 2-Dimensional matrix [output_len, input_len]
    :param input_tags: input token sequence
    :param output_tags: output token sequence
        if input_tags & output_tags aren't given, tick labels are number.
    :param title: figure title
    :return: None
    """
    output_len, input_len = attention_map.shape
    # print(attention_map.shape)
    # Plot the attention_map
    # plt.clf()
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(title)
    # Add image
    i = ax.matshow(attention_map, interpolation='nearest', cmap='Blues')

    # print(ax)
    # Add colorbar
    # cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    # cbar = f.colorbar(i, cax=cbaxes, orientation='vertical')
    # cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    font = fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')

    # Add labels
    ax.set_yticks(range(output_len))
    if output_tags is None:
        ax.set_yticklabels(list(range(output_len)))
    else:
        ax.set_yticklabels(output_tags[:output_len], fontproperties=font, fontsize='large')

    ax.set_xticks(range(input_len))
    if input_tags is None:
        ax.set_xticklabels(list(range(input_len)))
    else:
        ax.set_xticklabels(input_tags[:input_len], fontproperties=font, fontsize='large')

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()
    # plt.show()
    return
