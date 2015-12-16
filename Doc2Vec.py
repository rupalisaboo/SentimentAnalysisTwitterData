
# coding: utf-8

# In[43]:

get_ipython().run_cell_magic(u'bash', u'', u'#Split the all_views, get one file(sentiment, tweet) for each view\n#Clean each view\n#Run this for each view.\nnorm() {\n    fn=$1\n    if [ ! -f "$fn" ]\n    then\n        echo "File: $fn not found"\n        return 0\n    fi\n    #this function will convert text to lowercase and will disconnect punctuation and special symbols from words\n    function normalize_text {\n        awk \'{print tolower($0);}\' < $1 | sed -e \'s/\\./ \\. /g\' -e \'s/<br \\/>/ /g\' -e \'s/"/ " /g\' \\\n        -e \'s/,/ , /g\' -e \'s/(/ ( /g\' -e \'s/)/ ) /g\' -e \'s/\\!/ \\! /g\' -e \'s/\\?/ \\? /g\' \\\n        -e \'s/\\;/ \\; /g\' -e \'s/\\:/ \\: /g\' > $1-norm\n    }\n    export LC_ALL=C\n    normalize_text "$fn"\n    wc -l $fn\n    mv "$fn" "$fn-norm"\n}\nnorm "data/view1_clean" #file name is\nnorm "data/view2_clean"\nnorm "data/test_clean"')


# In[44]:

import os.path
tw_view_1 = 'data/view1_clean-norm'
tw_view_2 = 'data/view2_clean-norm'
tw_test = 'data/test_clean-norm'
assert os.path.isfile(tw_view_1), tw_view_1 + " unavailable"
assert os.path.isfile(tw_view_2), tw_view_2 + " unavailable"
assert os.path.isfile(tw_test), tw_test + " unavailable"


# In[45]:

from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
def stem_document(doc_sentence):
    words = doc_sentence.split()
    stemmed = ' '.join([stemmer.stem(word) for word in words])
    return stemmed


# In[46]:

import gensim
import numpy as np
#from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple, defaultdict as dd

#sentiment = {'positive':1, 'negative':-1} #, 'neutral':2}
sentiment_dict = {'4':1, '0':-1} #- new data 0,4
SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
stem = True

alldocs = dd(list)  # will hold all docs in original order - dictionary, keys = [v1, v2]
v1 = 'view1'
v2 = 'view2'
#tw_sentiment_dict = {}
#print total_num, train_test_shuffle
all_v2_words = []
with open(tw_view_2) as allview2:
        all_v2_words = allview2.readlines()
total_num = len(all_v2_words)
#split train/test
train_num = total_num *  9 / 10 # 70% train/test 1 - 10
train_test_shuffle = np.arange(total_num)
np.random.shuffle(train_test_shuffle)
with open(tw_view_1) as allview1:
    #for line_no, (v1, v2) in enumerate(zip(allview1, allview2)):
    for line_no, line in enumerate(allview1):
        tokens = gensim.utils.to_unicode(line).split('\t')
        if len(tokens) != 2:
            print line
            raise Exception()
        sentiment = sentiment_dict[tokens[0]]
        #if tw_id not in tw_sentiment_dict.keys():
        #    continue
        words = tokens[1]
        split = 'train' if train_test_shuffle[line_no] <= train_num else 'dev'
        #sentiment = tw_sentiment_dict[tw_id]
        v2_words = gensim.utils.to_unicode(all_v2_words[line_no]).split('\t')[1]
        
        alldocs[v1].append(SentimentDocument(stem_document(words) if stem else words,                                              ['%d_%s' %(line_no, v1)], split, sentiment))
        alldocs[v2].append(SentimentDocument(stem_document(v2_words) if stem else v2_words,                                              ['%d_%s' %(line_no, v2)], split, sentiment))
# test file
with open(tw_test) as test_fh:
    for line_no, line in enumerate(test_fh):
        tokens = gensim.utils.to_unicode(line).split('\t')
        if len(tokens) != 2:
            print line
            raise Exception()
        sentiment = sentiment_dict[tokens[0]]
        #if tw_id not in tw_sentiment_dict.keys():
        #    continue
        words = tokens[1]
        split = 'test'
        
        alldocs[v1].append(SentimentDocument(stem_document(words) if stem else words,                                              ['%d_%s' %(total_num + line_no, v1)], split, sentiment))
train_docs = {
    v1 : [doc for doc in alldocs[v1] if doc.split == 'train'],
    v2 : [doc for doc in alldocs[v2] if doc.split == 'train']
}
dev_docs = {
    v1 : [doc for doc in alldocs[v1] if doc.split == 'dev'],
    v2 : [doc for doc in alldocs[v2] if doc.split == 'dev']
}
test_docs = {
    v1 : [doc for doc in alldocs[v1] if doc.split == 'test']
}
doc_list = alldocs[v1][:] + alldocs[v2][:]  # for reshuffling per pass

print('%d total(view1 + view2) docs. view1: %d train, %d dev, %d test' % (len(doc_list), len(train_docs[v1]), len(dev_docs[v1]), len(test_docs[v1])))


# In[47]:

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"
model_size = 500

simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=model_size, window=3, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW 
    Doc2Vec(dm=0, size=model_size, negative=5, hs=0, min_count=5, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=model_size, window=3, negative=5, hs=0, min_count=2, workers=cores),
]

# speed setup by sharing results of 1st model's vocabulary scan
simple_models[0].build_vocab(doc_list)  # PV-DM/concat requires one special NULL word so it serves as template
print simple_models[0]
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print model

models_by_name = OrderedDict((str(model), model) for model in simple_models)


# In[48]:

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])
#print models_by_name['dbow+dmm'], models_by_name['dbow+dmc'] 
#del models_by_name['dbow+dmc']


# In[49]:

import numpy as np
import statsmodels.api as sm
from sklearn import svm, metrics, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from random import sample

# for timing
from contextlib import contextmanager
from timeit import default_timer
import time 
import ipdb

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
    
def logistic_predictor(train_targets, train_regressors):
    lr = LogisticRegression()
    lr.fit(train_regressors, train_targets)
    return lr

def svm_predictor(train_targets, train_regressors):
    svc = svm.SVC(kernel='rbf', degree=5, gamma=1e-1)
    svc.fit(train_regressors, train_targets)
    return svc

    """expected = svm_y_test
    predicted = svc.predict(svm_x_test)

    #print("Classification report for classifier %s:\n%s\n"
    #      % (svc, metrics.classification_report(expected, predicted)))
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    """
def rf_predictor(train_targets, train_regressors):
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(train_regressors, train_targets)
    return rfc

def error_rate_for_model(test_model, train_set, test_set,                          infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    predictor = predictor_alg(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps,                                                    alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_data]
    
    # predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    predicted = np.rint(test_predictions)
    expected = [doc.sentiment for doc in test_data]
    """if not infer:
        print("Classification report for classifier %s:\n%s\n"
              % (predictor, metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))"""
    #ipdb.set_trace()
    corrects = sum(expected == predicted)
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)


# In[50]:

from matplotlib import pyplot as plt
from scipy.linalg import eigh
get_ipython().magic(u'matplotlib inline')
def center(data):
    return data - np.mean(data, axis=0)

def PLS(X, Y):
    cross_cov = np.dot(center(X), center(Y).T)
    eigval,eigvec=np.linalg.eig(cross_cov.dot(cross_cov.T))
    return (eigval, eigvec)

def kcca(X, Y, regX=0.1, regY=0.1, numCC=10, kernelcca=True, ktype="gaussian"):
    '''Set up and solve the eigenproblem for the data in kernel and specified reg
    '''
    cenX = center(X)
    cenY = center(Y)
    kernel1 = np.array([_make_kernel(X.T, ktype=ktype)])
    kernel_x = (kernel1 + kernel1.T)/2
    kernel2 = np.array([_make_kernel(Y.T, ktype=ktype)])
    kernel_y = (kernel2 + kernel2.T)/2
    r_Ix = regX * np.eye(kernel_x.shape[0])
    r_Iy = regY * np.eye(kernel_y.shape[0])
    A = reduce(np.dot, [ np.linalg.inv(kernel_x - r_Ix), kernel_y, np.linalg.inv(kernel_y - r_Iy), kernel_x])
    eigval,eigvec=np.linalg.eig(A)
    return (eigval, eigvec)

def _listcorr(a):
    '''Returns pairwise row correlations for all items in array as a list of matrices
    '''
    corrs = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j > i:
                corrs[:, i, j] = [np.nan_to_num(np.corrcoef(ai, aj)[0, 1]) for (ai, aj) in zip(a[i].T, a[j].T)]
    return corrs


def recon(data, comp, corronly=False, kernelcca=True):
    nT = data[0].shape[0]
    # Get canonical variates and CCs
    if kernelcca:
        ws = _listdot(data, comp)
    else:
        ws = comp
    ccomp = _listdot([d.T for d in data], ws)
    corrs = _listcorr(ccomp)
    if corronly:
        return corrs
    else:
        return corrs, ws, ccomp


def _listdot(d1, d2): return [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]


def _make_kernel(d, normalize=True, ktype="linear", sigma=1.0):
    '''Makes a kernel for data d
      If ktype is "linear", the kernel is a linear inner product
      If ktype is "gaussian", the kernel is a Gaussian kernel with sigma = sigma
    '''
    if ktype == "linear":
        d = np.nan_to_num(d)
        cd = _demean(d)
        kernel = np.dot(cd, cd.T)
    elif ktype == "gaussian":
        from scipy.spatial.distance import pdist, squareform
        # this is an NxD matrix, where N is number of items and D its dimensionalites
        pairwise_dists = squareform(pdist(d, 'euclidean'))
        kernel = np.exp(-pairwise_dists ** 2 / sigma ** 2)
    kernel = (kernel + kernel.T) / 2.
    kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel


def _demean(d): return d - d.mean(0)

def CCA(X, Y, regX = 0, regY = 0):
    cenX = center(X)
    cenY = center(Y)
    cross_cov = cenX.dot(cenY.T)
    covX = cenX.dot(cenX.T)
    covY = cenY.dot(cenY.T)
    r_Ix = regX * np.eye(covX.shape[0])
    r_Iy = regY * np.eye(covY.shape[0])
    A = reduce(np.dot, [ np.linalg.inv(covX + r_Ix), cross_cov, np.linalg.inv(covY + r_Iy), cross_cov.T ])
    eigval,eigvec=np.linalg.eig(A)
    return (eigval, eigvec)

def get_top_eigvec(eigval, eigvec, k):
    idx=np.argsort(eigval)[-k:][::-1]
    #eigval=eigval[idx]
    return eigvec[:,idx]


# In[51]:

#from collections import defaultdict
best_error = dd(lambda: dd(lambda :(1.0, 0.0))) # { view: { model_name : (error_rate, alpha) } } ,to selectively-print only best errors achieved


# In[52]:

predictor_alg = logistic_predictor
from random import shuffle
import datetime

print 'Started.'
alpha, min_alpha, passes = (0.025, 0.001, 10)
alpha_delta = (alpha - min_alpha) / passes

print("START %s" % datetime.datetime.now())

for epoch in range(passes):
    shuffle(doc_list)  # shuffling gets best results

    for name, train_model in models_by_name.items():
        #print name
        # train
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            train_model.train(doc_list)
            duration = '%.1f' % elapsed()

        #print np.array(train_model.docvecs[['%d_%s' %(0, view)]]).shape
        # evaluate
        #view1, view2
        for view in [v1, v2]:
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs[view], dev_docs[view])
            eval_duration = '%.1f' % eval_elapsed()
            best_indicator = ' '
            if err < best_error[view][name][0]:
                best_error[view][name] = (err, alpha)
                best_indicator = '*' 
            #print("%s%f : %i passes : %s-%s %ss %ss" % (best_indicator, err, epoch + 1, view, name, duration, eval_duration))

    #print('completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta

print("END %s" % str(datetime.datetime.now()))


# In[53]:

for view in [v1, v2]:
    print '========= %s ========' %view
    for rate, alpha, name in sorted((rate, alpha, name) for name, (rate, alpha) in best_error[view].items()):
        print("%f %s %f" % (rate, name, alpha))


# In[54]:

doc_id = np.random.randint(len(train_docs[v2]) + len(dev_docs[v2]))  # pick random doc; re-run cell for more examples
print('for doc %d...' % doc_id)
# Print example tweet and vector reps for both views
print alldocs[v1][doc_id]
#tag = alldocs['view1'][doc_id].tags[0]
#print '\n', simple_models['view1'][0].docvecs[tag]

print '\n', alldocs[v2][doc_id]
#print '\n', simple_models['view2'][0].docvecs[tag]
#print '\n\n', doc_list['view2'][:10]
for model in simple_models:
    inferred_docvec = model.infer_vector(alldocs[v1][doc_id].words)
    print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))


# In[55]:

#Select the best performing word2vec model
_, best_alpha, best_model_name = min(((rate, alpha, name)                                            for name, (rate, alpha) in best_error[v1].items()), key=lambda b: b[0])
print best_model_name 
print best_alpha
best_model = models_by_name[best_model_name]

# Train best model
shuffle(doc_list)
best_model.alpha, best_model.min_alpha = best_alpha, best_alpha
best_model.train(doc_list)


# In[56]:

# DO CCA on the training docvecs
# X = view 1, Y = view 2 : [word_vec_size x num_samples]
target_sentiments, X, Y = zip(*[(doc.sentiment, best_model.docvecs[doc.tags[0]],                              best_model.docvecs[doc.tags[0].replace(v1, v2)]) for doc in train_docs[v1]])
X = np.asarray(X).T
Y = np.asarray(Y).T
#dev docs
dev = [best_model.docvecs[doc.tags[0]] for doc in dev_docs[v1]]
dev = np.asarray(dev).T
#test docs
test = [best_model.docvecs[doc.tags[0]] for doc in test_docs[v1]]
test = np.asarray(test).T


# In[57]:

(cca_eigval, cca_eigvec) = CCA(X, Y) #PLS(X, Y)  #kcca(X, Y, regX=0.1, regY=0.1, numCC=10, kernelcca=True, ktype="gaussian")
print np.shape(X), np.shape(Y)
print np.shape(cca_eigvec)
#print np.transpose(X)


# In[92]:

def error_rate(X, X_targets, test, expected, print_conf=False):
    predictor = predictor_alg(X_targets, X)

    # predict & evaluate
    predictions = predictor.predict(test)
    predicted = np.rint(predictions)
    #print("Classification report for %s:\n%s\n" % (predictor, metrics.classification_report(expected, predicted)))
    if print_conf:
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    #ipdb.set_trace()
    errors = len(predictions) - sum(expected == predicted)
    err_orig = float(errors) / len(expected)
    if print_conf:
        print 1-err_orig
    return (1-err_orig)*100


# In[93]:

expected = [doc.sentiment for doc in dev_docs[v1]]
err_orig = error_rate(X.T, target_sentiments, dev.T, expected, True)


# In[103]:

#get top k eigvec, project training data and stack with original word vectors
err_cca = []
step = model_size/25
num_dir_ranges = range(step, step+model_size, step)
for num_dir in num_dir_ranges:
    top_k_eigv = get_top_eigvec(cca_eigval, cca_eigvec, num_dir)
    #print np.shape(top_k_eigv)
    X_proj = top_k_eigv.T.dot(X)
    #print np.shape(X_proj)
    stacked_vec = np.append(X, X_proj, axis=0)
    #print np.shape(stacked_vec)

    # project test data to cca directions and stack
    #print np.shape(test)
    dev_proj = top_k_eigv.T.dot(dev)
    stacked_dev = np.append(dev, dev_proj, axis=0)
    #print np.shape(stacked_dev)
    #print np.shape(target_sentiments)
    
    expected = [doc.sentiment for doc in dev_docs[v1]]
    err_cca.append(error_rate(stacked_vec.T, target_sentiments, stacked_dev.T, expected))


# In[104]:

plt.plot(num_dir_ranges, [err_orig]*len(num_dir_ranges), 'r--', label='Paragraph vec.')
plt.plot(num_dir_ranges, err_cca, 'b*', label='Paragraph+CCA')
plt.ylabel('Classification Accuracy(%)')
plt.xlabel('Number of CCA directions stacked with paragraph vectors')
plt.grid()
plt.legend(loc='best')
plt.show()


# In[62]:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
corpus = [doc.words for doc in alldocs[v1]]
tf_idf = vectorizer.fit_transform(corpus).toarray()


# In[63]:

#np.shape(tf_idf)
tf_X = tf_idf[np.array([int(doc.tags[0].replace('_'+v1, '')) for doc in train_docs[v1]])]
tf_dev = tf_idf[np.array([int(doc.tags[0].replace('_'+v1, '')) for doc in dev_docs[v1]])]
tf_test = tf_idf[np.array([int(doc.tags[0].replace('_'+v1, '')) for doc in test_docs[v1]])]


# In[105]:

expected = [doc.sentiment for doc in dev_docs[v1]]
#TF-IDF
err_tf = error_rate(tf_X, target_sentiments,            tf_dev, expected, True)
#Tf-IDF stacked with CCA
err_tf_cca = []
step = model_size/25
num_dir_ranges = range(step, step+model_size, step)
for num_dir in num_dir_ranges:
    top_k_eigv = get_top_eigvec(cca_eigval, cca_eigvec, num_dir)
    #print np.shape(top_k_eigv)
    X_proj = top_k_eigv.T.dot(X)
    #print np.shape(X_proj)
    #stacked_vec = np.append(X, X_proj, axis=0)
    #print np.shape(stacked_vec)

    # project test data to cca directions and stack
    #print np.shape(test)
    dev_proj = top_k_eigv.T.dot(dev)
    #stacked_dev = np.append(dev, dev_proj, axis=0)
    #print np.shape(stacked_dev)
    #print np.shape(target_sentiments)
    err_tf_cca.append(error_rate(np.append(tf_X, X_proj.T, axis=1), target_sentiments,                np.append(tf_dev, dev_proj.T, axis=1), expected))


# In[106]:

#TF-IDF stacked with doc2vec
expected = [doc.sentiment for doc in dev_docs[v1]]
err_tf_doc = error_rate(np.append(tf_X, X.T, axis=1), target_sentiments,                np.append(tf_dev, dev.T, axis=1), expected, True)


# In[107]:

plt.plot(num_dir_ranges, [err_tf]*len(num_dir_ranges), 'r--', label='TF-IDF')
plt.plot(num_dir_ranges, [err_tf_doc]*len(num_dir_ranges), 'g--', label='TF-IDF+Paragraph')
plt.plot(num_dir_ranges, err_tf_cca, 'b*', label='TF-IDF+CCA')
plt.ylabel('Classification Accuracy(%)')
plt.xlabel('Number of CCA directions stacked with TF-IDF vectors')
plt.grid()
plt.legend(loc='best')
plt.show()


# In[108]:

#Test - TF-idf
expected = [doc.sentiment for doc in test_docs[v1]]
error_rate(tf_X, target_sentiments, tf_test, expected, True)
#test - Doc2Vec stacked with tf-idf
error_rate(np.append(tf_X, X.T, axis=1), target_sentiments, np.append(tf_test, test.T, axis=1), expected, True)


# In[111]:

#Test tf-idf stacked with cca
n = 50
top_k_eigv = get_top_eigvec(cca_eigval, cca_eigvec, n)

X_proj = top_k_eigv.T.dot(X)

test_proj = top_k_eigv.T.dot(test)

error_rate(np.append(tf_X, X_proj.T, axis=1), target_sentiments,            np.append(tf_test, test_proj.T, axis=1), expected, True)


# In[110]:

print err_tf_cca


# In[ ]:



