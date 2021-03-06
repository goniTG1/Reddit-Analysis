# -*- coding: utf-8 -*-
"""API_reddit.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1D8IEuHkG_PU3qYc1abHIU3JM7BCxRg8l
# INSTALL & IMPORT LIBRARIES
"""

#!pip install --upgrade spacy
#!python -m spacy download en_core_web_md
#!pip install googletrans


import os       #importing os to set environment variable
"""
def install_java():
    !apt-get install -y openjdk-8-jdk-headless -qq > /dev/null      #install openjdk
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"     #set environment variable
    !java -version       #check java version
install_java()
"""


#!wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
#!unzip mallet-2.0.8.zip

os.environ['MALLET_HOME'] = 'mallet-2.0.8'
mallet_path = 'mallet-2.0.8/bin/mallet' # you should NOT need to change this 

#!pip install --upgrade gensim==3.8.3
#!pip install pyLDAvis==2.1.2

import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored
import seaborn as sns
from textblob import TextBlob

import spacy
import gensim
from gensim.models.phrases import Phrases
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel

import pyLDAvis.gensim as gensimvis
import pyLDAvis

from sklearn import linear_model

LANGUAGES = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'he': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'or': 'odia',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'tk': 'turkmen',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'ug': 'uyghur',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu',
}

DEFAULT_SERVICE_URLS = ('translate.google.ac','translate.google.ad','translate.google.ae',
                        'translate.google.al','translate.google.am','translate.google.as',
                        'translate.google.at','translate.google.az','translate.google.ba',
                        'translate.google.be','translate.google.bf','translate.google.bg',
                        'translate.google.bi','translate.google.bj','translate.google.bs',
                        'translate.google.bt','translate.google.by','translate.google.ca',
                        'translate.google.cat','translate.google.cc','translate.google.cd',
                        'translate.google.cf','translate.google.cg','translate.google.ch',
                        'translate.google.ci','translate.google.cl','translate.google.cm',
                        'translate.google.cn','translate.google.co.ao','translate.google.co.bw',
                        'translate.google.co.ck','translate.google.co.cr','translate.google.co.id',
                        'translate.google.co.il','translate.google.co.in','translate.google.co.jp',
                        'translate.google.co.ke','translate.google.co.kr','translate.google.co.ls',
                        'translate.google.co.ma','translate.google.co.mz','translate.google.co.nz',
                        'translate.google.co.th','translate.google.co.tz','translate.google.co.ug',
                        'translate.google.co.uk','translate.google.co.uz','translate.google.co.ve',
                        'translate.google.co.vi','translate.google.co.za','translate.google.co.zm',
                        'translate.google.co.zw','translate.google.co','translate.google.com.af',
                        'translate.google.com.ag','translate.google.com.ai','translate.google.com.ar',
                        'translate.google.com.au','translate.google.com.bd','translate.google.com.bh',
                        'translate.google.com.bn','translate.google.com.bo','translate.google.com.br',
                        'translate.google.com.bz','translate.google.com.co','translate.google.com.cu',
                        'translate.google.com.cy','translate.google.com.do','translate.google.com.ec',
                        'translate.google.com.eg','translate.google.com.et','translate.google.com.fj',
                        'translate.google.com.gh','translate.google.com.gi','translate.google.com.gt',
                        'translate.google.com.hk','translate.google.com.jm','translate.google.com.kh',
                        'translate.google.com.kw','translate.google.com.lb','translate.google.com.lc',
                        'translate.google.com.ly','translate.google.com.mm','translate.google.com.mt',
                        'translate.google.com.mx','translate.google.com.my','translate.google.com.na',
                        'translate.google.com.ng','translate.google.com.ni','translate.google.com.np',
                        'translate.google.com.om','translate.google.com.pa','translate.google.com.pe',
                        'translate.google.com.pg','translate.google.com.ph','translate.google.com.pk',
                        'translate.google.com.pr','translate.google.com.py','translate.google.com.qa',
                        'translate.google.com.sa','translate.google.com.sb','translate.google.com.sg',
                        'translate.google.com.sl','translate.google.com.sv','translate.google.com.tj',
                        'translate.google.com.tr','translate.google.com.tw','translate.google.com.ua',
                        'translate.google.com.uy','translate.google.com.vc','translate.google.com.vn',
                        'translate.google.com','translate.google.cv','translate.google.cx',
                        'translate.google.cz','translate.google.de','translate.google.dj',
                        'translate.google.dk','translate.google.dm','translate.google.dz',
                        'translate.google.ee','translate.google.es','translate.google.eu',
                        'translate.google.fi','translate.google.fm','translate.google.fr',
                        'translate.google.ga','translate.google.ge','translate.google.gf',
                        'translate.google.gg','translate.google.gl','translate.google.gm',
                        'translate.google.gp','translate.google.gr','translate.google.gy',
                        'translate.google.hn','translate.google.hr','translate.google.ht',
                        'translate.google.hu','translate.google.ie','translate.google.im',
                        'translate.google.io','translate.google.iq','translate.google.is',
                        'translate.google.it','translate.google.je','translate.google.jo',
                        'translate.google.kg','translate.google.ki','translate.google.kz',
                        'translate.google.la','translate.google.li','translate.google.lk',
                        'translate.google.lt','translate.google.lu','translate.google.lv',
                        'translate.google.md','translate.google.me','translate.google.mg',
                        'translate.google.mk','translate.google.ml','translate.google.mn',
                        'translate.google.ms','translate.google.mu','translate.google.mv',
                        'translate.google.mw','translate.google.ne','translate.google.nf',
                        'translate.google.nl','translate.google.no','translate.google.nr',
                        'translate.google.nu','translate.google.pl','translate.google.pn',
                        'translate.google.ps','translate.google.pt','translate.google.ro',
                        'translate.google.rs','translate.google.ru','translate.google.rw',
                        'translate.google.sc','translate.google.se','translate.google.sh',
                        'translate.google.si','translate.google.sk','translate.google.sm',
                        'translate.google.sn','translate.google.so','translate.google.sr',
                        'translate.google.st','translate.google.td','translate.google.tg',
                        'translate.google.tk','translate.google.tl','translate.google.tm',
                        'translate.google.tn','translate.google.to','translate.google.tt',
                        'translate.google.us','translate.google.vg','translate.google.vu','translate.google.ws')

import json, requests, random, re
from urllib.parse import quote
import urllib3
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URLS_SUFFIX = [re.search('translate.google.(.*)', url.strip()).group(1) for url in DEFAULT_SERVICE_URLS]
URL_SUFFIX_DEFAULT = 'cn'


class google_new_transError(Exception):
    """Exception that uses context to present a meaningful error message"""

    def __init__(self, msg=None, **kwargs):
        self.tts = kwargs.pop('tts', None)
        self.rsp = kwargs.pop('response', None)
        if msg:
            self.msg = msg
        elif self.tts is not None:
            self.msg = self.infer_msg(self.tts, self.rsp)
        else:
            self.msg = None
        super(google_new_transError, self).__init__(self.msg)

    def infer_msg(self, tts, rsp=None):
        cause = "Unknown"

        if rsp is None:
            premise = "Failed to connect"

            return "{}. Probable cause: {}".format(premise, "timeout")
            # if tts.tld != 'com':
            #     host = _translate_url(tld=tts.tld)
            #     cause = "Host '{}' is not reachable".format(host)

        else:
            status = rsp.status_code
            reason = rsp.reason

            premise = "{:d} ({}) from TTS API".format(status, reason)

            if status == 403:
                cause = "Bad token or upstream API changes"
            elif status == 200 and not tts.lang_check:
                cause = "No audio stream in response. Unsupported language '%s'" % self.tts.lang
            elif status >= 500:
                cause = "Uptream API error. Try again later."

        return "{}. Probable cause: {}".format(premise, cause)

class google_translator:

    def __init__(self, url_suffix="cn", timeout=5, proxies=None):
        self.proxies = proxies
        if url_suffix not in URLS_SUFFIX:
            self.url_suffix = URL_SUFFIX_DEFAULT
        else:
            self.url_suffix = url_suffix
        url_base = "https://translate.google.{}".format(self.url_suffix)
        self.url = url_base + "/_/TranslateWebserverUi/data/batchexecute"
        self.timeout = timeout

    def _package_rpc(self, text, lang_src='auto', lang_tgt='auto'):
        GOOGLE_TTS_RPC = ["MkEWBc"]
        parameter = [[text.strip(), lang_src, lang_tgt, True], [1]]
        escaped_parameter = json.dumps(parameter, separators=(',', ':'))
        rpc = [[[random.choice(GOOGLE_TTS_RPC), escaped_parameter, None, "generic"]]]
        espaced_rpc = json.dumps(rpc, separators=(',', ':'))
        # text_urldecode = quote(text.strip())
        freq_initial = "f.req={}&".format(quote(espaced_rpc))
        freq = freq_initial
        return freq

    def translate(self, text, lang_tgt='auto', lang_src='auto', pronounce=False):
        try:
            lang = LANGUAGES[lang_src]
        except:
            lang_src = 'auto'
        try:
            lang = LANGUAGES[lang_tgt]
        except:
            lang_src = 'auto'
        text = str(text)
        if len(text) >= 5000:
            return "Warning: Can only detect less than 5000 characters"
        if len(text) == 0:
            return ""
        headers = {
            "Referer": "http://translate.google.{}/".format(self.url_suffix),
            "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; WOW64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/47.0.2526.106 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        freq = self._package_rpc(text, lang_src, lang_tgt)
        response = requests.Request(method='POST',
                                    url=self.url,
                                    data=freq,
                                    headers=headers,
                                    )
        try:
            if self.proxies == None or type(self.proxies) != dict:
                self.proxies = {}
            with requests.Session() as s:
                s.proxies = self.proxies
                r = s.send(request=response.prepare(),
                           verify=False,
                           timeout=self.timeout)
            for line in r.iter_lines(chunk_size=1024):
                decoded_line = line.decode('utf-8')
                if "MkEWBc" in decoded_line:
                    try:
                        response = decoded_line
                        response = json.loads(response)
                        response = list(response)
                        response = json.loads(response[0][2])
                        response_ = list(response)
                        response = response_[1][0]
                        if len(response) == 1:
                            if len(response[0]) > 5:
                                sentences = response[0][5]
                            else: ## only url
                                sentences = response[0][0]
                                if pronounce == False:
                                    return sentences
                                elif pronounce == True:
                                    return [sentences,None,None]
                            translate_text = ""
                            for sentence in sentences:
                                sentence = sentence[0]
                                translate_text += sentence.strip() + ' '
                            translate_text = translate_text
                            if pronounce == False:
                                return translate_text
                            elif pronounce == True:
                                pronounce_src = (response_[0][0])
                                pronounce_tgt = (response_[1][0][0][1])
                                return [translate_text, pronounce_src, pronounce_tgt]
                        elif len(response) == 2:
                            sentences = []
                            for i in response:
                                sentences.append(i[0])
                            if pronounce == False:
                                return sentences
                            elif pronounce == True:
                                pronounce_src = (response_[0][0])
                                pronounce_tgt = (response_[1][0][0][1])
                                return [sentences, pronounce_src, pronounce_tgt]
                    except Exception as e:
                        raise e
            r.raise_for_status()
        except requests.exceptions.ConnectTimeout as e:
            raise e
        except requests.exceptions.HTTPError as e:
            # Request successful, bad response
            raise google_new_transError(tts=self, response=r)
        except requests.exceptions.RequestException as e:
            # Request failed
            raise google_new_transError(tts=self)

    def detect(self, text):
        text = str(text)
        if len(text) >= 5000:
            return log.debug("Warning: Can only detect less than 5000 characters")
        if len(text) == 0:
            return ""
        headers = {
            "Referer": "http://translate.google.{}/".format(self.url_suffix),
            "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; WOW64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/47.0.2526.106 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        freq = self._package_rpc(text)
        response = requests.Request(method='POST',
                                    url=self.url,
                                    data=freq,
                                    headers=headers)
        try:
            if self.proxies == None or type(self.proxies) != dict:
                self.proxies = {}
            with requests.Session() as s:
                s.proxies = self.proxies
                r = s.send(request=response.prepare(),
                           verify=False,
                           timeout=self.timeout)

            for line in r.iter_lines(chunk_size=1024):
                decoded_line = line.decode('utf-8')
                if "MkEWBc" in decoded_line:
                    # regex_str = r"\[\[\"wrb.fr\",\"MkEWBc\",\"\[\[(.*).*?,\[\[\["
                    try:
                        # data_got = re.search(regex_str,decoded_line).group(1)
                        response = (decoded_line + ']')
                        response = json.loads(response)
                        response = list(response)
                        response = json.loads(response[0][2])
                        response = list(response)
                        detect_lang = response[0][2]
                    except Exception:
                        raise Exception
                    # data_got = data_got.split('\\\"]')[0]
                    return [detect_lang, LANGUAGES[detect_lang.lower()]]
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Request successful, bad response
            log.debug(str(e))
            raise google_new_transError(tts=self, response=r)
        except requests.exceptions.RequestException as e:
            # Request failed
            log.debug(str(e))
            raise google_new_transError(tts=self)

"""# DATA GATHERING"""


def get_df(sel_type, subreddit):

    CLIENT_ID = '0LlqT7VnljEp8DI1zAbd7Q'
    SECRET_KEY = 'DJTVPIv6-Ans8DUGgLkgKrqYa-7eGA'
    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET_KEY)
    data = {'grant_type': 'password','username': 'richiestaire','password': 'jarak#5#2021',}
    headers = {'User-Agent':'MyAPI/0.0.1'}
    res = requests.post('https://www.reddit.com/api/v1/access_token',auth=auth, data=data, headers=headers)
    token = res.json()['access_token']
    headers['Authorization'] = f'bearer {token}'

    res = requests.get(f'https://oauth.reddit.com/r/{subreddit}/{sel_type}', headers = headers, params = {'limit':'50', 't':'year'})

    df = pd.DataFrame({'post id': [], 'author': [], 'name': [], 'title': [], 'year': [], 'month': [], 'day': [], 'selftext': [], 'score': [], 'upvote ratio': [],
                       '# comments': [], 'media': [], 'media only': [], 'video': [], 'gilded': [],'distinguised':[], 'clicked': [],
                      'downs': [], 'total_awards_received':[], 'ups': [], 'pinned':[],  'num_crossposts': []})

    for post in res.json()['data']['children']:
      df = df.append({
        'post id': post['data']['id'],
        'author' : post['data']['author'],
        'name' : post['data']['name'],
        'title' : post['data']['title'],
        'year': datetime.datetime.utcfromtimestamp(post['data']['created_utc']).year,
        'month': datetime.datetime.utcfromtimestamp(post['data']['created_utc']).month,
        'day': datetime.datetime.utcfromtimestamp(post['data']['created_utc']).day,
        'selftext' : post['data']['selftext'],
        'score' : post['data']['score'],
        'upvote ratio': post['data']['upvote_ratio'],
        '# comments' : post['data']['num_comments'],
        'media': post['data']['media'],
        'media only': post['data']['media_only'],
        'video': post['data']['is_video'],
        'gilded': post['data']['gilded'],
        'distinguised': post['data']['distinguished'],
        'clicked': post['data']['clicked'],
        'downs': post['data']['downs'],
        'ups': post['data']['ups'],
        'total_awards_received': post['data']['total_awards_received'],
        'pinned': post['data']['pinned'],
        'num_crossposts': post['data']['num_crossposts']
        }, ignore_index=True)

    """
    while True:
      res = requests.get(f'https://oauth.reddit.com/r/{subreddit}/{sel_type}', headers = headers, params = {'limit':'100', 'after': df['name'].iloc[len(df) - 1], 't': 'year'})
      if len(res.json()['data']['children']) == 0:
        break
      for post in res.json()['data']['children']:
        df = df.append({
          'post id': post['data']['id'],
          'author' : post['data']['author'],
          'name' : post['data']['name'],
          'title' : post['data']['title'],
          'year': datetime.datetime.utcfromtimestamp(post['data']['created_utc']).year,
          'month': datetime.datetime.utcfromtimestamp(post['data']['created_utc']).month,
          'day': datetime.datetime.utcfromtimestamp(post['data']['created_utc']).day,
          'selftext' : post['data']['selftext'],
          'score' : post['data']['score'],
          'upvote ratio': post['data']['upvote_ratio'],
          '# comments' : post['data']['num_comments'],
          'media': post['data']['media'],
          'media only': post['data']['media_only'],
          'video': post['data']['is_video']}, ignore_index=True)
    """
    return filter_df(df, subreddit, headers)


def filter_df(df, subreddit, headers):

    postsids = df['post id'].to_list()
    post_comments = []
    for i in postsids:
      comments = requests.get(f'https://oauth.reddit.com/r/{subreddit}/comments/{i}/', headers = headers, params = {'limit':'100'})
      c = []
      for i in range(len(comments.json()[1]['data']['children'])):
        try:
          c.append(comments.json()[1]['data']['children'][i]['data']['body'])
        except:
          pass
      post_comments.append(c)

    df['Comments'] = post_comments

    """# TRANSLATE TITLES & COMMENTS FOR NLP"""

    translator = google_translator()

    if subreddit == 'London' or subreddit == 'Dublin':
        df['Eng Title'] = df['title'].astype(str)
        df['Eng Comments'] = df['Comments'].astype(str)

    elif subreddit != 'london':
        eng_titles = []; eng_comments = []
        for i in range(len(df)):
            translate_text = translator.translate(df['title'][i])
            eng_titles.append(translate_text)
            translate_comment = translator.translate(df['Comments'][i])
            eng_comments.append(translate_comment)
        df['Eng Title'] = eng_titles; df['Eng Comments'] = eng_comments

    c = []
    for i in range(len(df)):
        a = re.sub(r'[^A-Za-z0-9 ]+', '', df['Eng Comments'][i])
        c.append(a)
    df['Eng Comments'] = c

    """# FILTER POSTS BY MONTH"""

    m_selected = [1.0,2.0,3.0] #months selected by user
    all_m = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]

    for i in m_selected:
      if i in all_m:
        all_m.remove(i)
    """# NATURAL LANGUAGE  PROCESSING
    """
    filter_df = df

    nlp = spacy.load('en_core_web_md')
    nlp.max_length = 10000000
    nlp = spacy.load('en_core_web_md')
    nlp.disable_pipe('parser')
    nlp.disable_pipe('ner')
    valid_POS = set(['VERB', 'NOUN', 'ADJ', 'PROPN'])
    filter_df["New_Title"] = filter_df['Eng Title'].apply(lambda x: " ".join([y.lemma_ for y in nlp(x)]))
    filter_df["New_Title"] = filter_df['New_Title'].apply(
        lambda x: " ".join([token.lemma_ for token in nlp(x) if token.is_alpha and token.pos_ in valid_POS
                            and token.is_stop == False]))

    filter_df['New_Title'] = filter_df.New_Title.astype(str)
    filter_df = filter_df.reset_index(drop=True)
    Final_Title_l = []
    for i in range(len(filter_df['New_Title'])):
        pepe = filter_df['New_Title'][i].lower()
        Final_Title_l.append(pepe)

    filter_df['New_Title'] = Final_Title_l

    # Gensim Part
    mycorpus = [gensim.utils.simple_preprocess(line, deacc=True) for line in filter_df.New_Title]
    phrase_model = Phrases(mycorpus, min_count=1, threshold=20)
    mycorpus = [el for el in phrase_model[mycorpus]]  # We populate mycorpus again

    Ngrams = {}
    for doc in mycorpus:
        for token in doc:
            if token in Ngrams:
                Ngrams[token] += 1
            else:
                Ngrams[token] = 1
    Ngrams_df = pd.DataFrame(Ngrams.items(), columns=['Ngrams', 'count'])
    Ngrams_df.set_index('Ngrams')
    Ngrams_df = Ngrams_df.sort_values(by=['count'], ascending=False).reset_index(drop=True)

    topic = []
    for j in range(len(filter_df['New_Title'])):
        counts = []
        if len(filter_df['New_Title'][j].split(' ')) > 0:
            for i in filter_df['New_Title'][j].split(' '):
                if i in Ngrams_df['Ngrams'].to_list():
                    counts.append(int(Ngrams_df[Ngrams_df['Ngrams'] == i]['count']))
            if len(counts) > 0:
                m = counts.index(max(counts))
                topic.append(filter_df['New_Title'][j].split(' ')[m])
            else:
                topic.append('')
        else:
            topic.append('')

    filter_df['Topic'] = topic

    def senti(x):
        return TextBlob(x).sentiment
    sentim = []
    for i in range(len(filter_df)):
        sentim.append(senti(filter_df['Eng Comments'][i]))

    polarity = []
    subjectivity = []

    for i in range(len(sentim)):
        polarity.append(sentim[i][0])
        subjectivity.append(sentim[i][1])

    filter_df['Polarity'] = polarity
    filter_df['Subjectivity'] = subjectivity

    return filter_df


def most_frequent(List):
    return max(set(List), key=List.count)
def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

def get_popular_values(filter_df, n, subreddit):
    #get most repeated topics
    topic = filter_df['Topic'].to_list()
    topic = [elem for elem in topic if str(elem).lower() != subreddit.lower()]
    mf_topics = []
    for i in range((n)):
        mf = most_frequent(topic)
        while len(mf) == 0:
            topic = remove_values_from_list(topic, mf)
            mf = most_frequent(topic)
            mf_topics.append(mf)
            topic = remove_values_from_list(topic, mf)
        else:
            if mf not in mf_topics:
                mf_topics.append(mf)
                topic = remove_values_from_list(topic, mf)

    months = filter_df['month'].value_counts()
    w = 0; s = 0; su = 0; f = 0
    sea = ['Winter', 'Spring', 'Summer', 'Fall']
    k = 0
    for i in months.keys():
        if i == 1.0 or i == 2.0 or i == 3.0:
            w += months.values[k]
        elif i == 4.0 or i == 5.0 or i == 6.0:
            s += months.values[k]
        elif i == 7.0 or i == 8.0 or i == 9.0:
            su += months.values[k]
        else:
            f += months.values[k]
        k += 1

    month = numToMonth(months.keys()[0])
    num = [w,s,su,f]
    return num, [month, months.keys()[0], months], sea, mf_topics


def numToMonth(month_value):
    return {
            1: 'January',
            2: 'February',
            3: 'March',
            4: 'April',
            5: 'May',
            6: 'June',
            7: 'July',
            8: 'August',
            9: 'September',
            10: 'October',
            11: 'November',
            12: 'December'
    }[month_value]