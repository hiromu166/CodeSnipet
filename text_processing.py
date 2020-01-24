import re
import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas(desc="hoge progress: ")

def delete_brackets(s):
    """
    括弧と括弧内文字列を削除
    """
    """ brackets to zenkaku """
    table = {
        "(": "（",
        ")": "）",
        "<": "＜",
        ">": "＞",
        "{": "｛",
        "}": "｝",
        "[": "［",
        "]": "］"
    }
    for key in table.keys():
        s = s.replace(key, table[key])
    """ delete zenkaku_brackets """
    l = ['（[^（|^）]*）', '【[^【|^】]*】', '＜[^＜|^＞]*＞', '［[^［|^］]*］',
         '「[^「|^」]*」', '｛[^｛|^｝]*｝', '〔[^〔|^〕]*〕', '〈[^〈|^〉]*〉']
    for l_ in l:
        s = re.sub(l_, "", s)
    """ recursive processing """
    return delete_brackets(s) if sum([1 if re.search(l_, s) else 0 for l_ in l]) > 0 else s

def tokenize_text(df, col):
    df[col] = df[col].progress_apply(lambda x: delete_brackets(x))
    t_wakati = Tokenizer(wakati=True)
    df[col] = df[col].progress_apply(lambda x: ' '.join(t_wakati.tokenize(x)))
    return df[col]

def TFIDF(df, col, ngram=(1,1)):
    tfidf_vec = TfidfVectorizer(ngram_range=ngram, max_features=None)
    tfidf_vector = tfidf_vec.fit_transform(df[col])
    return tfidf_vector

def SVD(df, col, n_components, tfidf_vector):
    svd_vec = TruncatedSVD(n_components=n_components, algorithm='arpack')
    svd_vec.fit(tfidf_vector)
    svd_columns = pd.DataFrame(svd_vec.transform(tfidf_vector))
    svd_columns.columns = ['svd_%s_%d' % (col, x) for x in range(n_components)]
    return svd_columns
