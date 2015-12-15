
# coding: utf-8

# In[846]:

get_ipython().run_cell_magic(u'bash', u'', u'#Split the all_views, get one file(sentiment, tweet) for each view\n#Clean each view\n#Run this for each view.\nnorm() {\n    fn=$1\n    if [ ! -f "$fn" ]\n    then\n        echo "File: $fn not found"\n        return 0\n    fi\n    #this function will convert text to lowercase and will disconnect punctuation and special symbols from words\n    function normalize_text {\n        awk \'{print tolower($0);}\' < $1 | sed -e \'s/\\./ \\. /g\' -e \'s/<br \\/>/ /g\' -e \'s/"/ " /g\' \\\n        -e \'s/,/ , /g\' -e \'s/(/ ( /g\' -e \'s/)/ ) /g\' -e \'s/\\!/ \\! /g\' -e \'s/\\?/ \\? /g\' \\\n        -e \'s/\\;/ \\; /g\' -e \'s/\\:/ \\: /g\' > $1-norm\n    }\n    export LC_ALL=C\n    normalize_text "$fn"\n    wc -l $fn\n    mv "$fn" "$fn-norm"\n}\nnorm "data/view1_clean" #file name is\nnorm "data/view2_clean"\nnorm "data/test_clean"')


# In[847]:

import os.path
tw_view_1 = 'data/view1_clean-norm'
tw_view_2 = 'data/view2_clean-norm'
test = 'data/test_clean-norm'
assert os.path.isfile(tw_view_1), tw_view_1 + " unavailable"
assert os.path.isfile(tw_view_2), tw_view_2 + " unavailable"
assert os.path.isfile(test), test + " unavailable"


# In[823]:

from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
def stem_document(doc_sentence):
    words = doc_sentence.split()
    stemmed = ' '.join([stemmer.stem(word) for word in words])
    return stemmed


# In[849]:

import gensim
import numpy as np
from gensim.models.doc2vec import TaggedDocument
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
train_num = total_num *  7 / 10 # 70% train/test 1 - 10
train_test_shuffle = np.arange(total_num)
np.random.shuffle(train_test_shuffle)
with open(tw_view_1) as allview1:
    #for line_no, (v1, v2) in enumerate(zip(allview1, allview2)):
    for line_no, line in enumerate(allview1):
        tokens = gensim.utils.to_unicode(line.strip()).split('\t')
        if len(tokens) != 2:
            print line
            raise Exception()
        sentiment = sentiment_dict[tokens[0]]
        #if tw_id not in tw_sentiment_dict.keys():
        #    continue
        words = tokens[1]
        split = 'train' if train_test_shuffle[line_no] <= train_num else 'dev'
        #sentiment = tw_sentiment_dict[tw_id]
        v2_words = gensim.utils.to_unicode(all_v2_words[line_no].strip()).split('\t')[1]
        
        alldocs[v1].append(SentimentDocument(stem_document(words) if stem else words, [line_no], split, sentiment))
        alldocs[v2].append(SentimentDocument(stem_document(v2_words) if stem else v2_words, [line_no], split, sentiment))
train_docs = {
    v1 : [doc for doc in alldocs[v1] if doc.split == 'train'],
    v2 : [doc for doc in alldocs[v2] if doc.split == 'train']
}
dev_docs = {
    v1 : [doc for doc in alldocs[v1] if doc.split == 'dev'],
    v2 : [doc for doc in alldocs[v2] if doc.split == 'dev']
}
doc_list = { v1: alldocs[v1][:], v2: alldocs[v2][:] }  # for reshuffling per pass

print('%d docs: %d train-sentiment, %d dev-sentiment' % (len(doc_list[v1]), len(train_docs[v1]), len(test_docs[v1])))


# # get test data
# test_tweets = []
# with open(tw_test) as test_fh:
#     for line_no, line in enumerate(test_fh):
#         tokens = gensim.utils.to_unicode(line.strip()).split('\t')
#         if len(tokens) != 2:
#             print line
#             raise Exception()
#         test_tweets.append((sentiment_dict[tokens[0]], stem_document(tokens[1]) if stem else tokens[1]))
# print '%d test docs' %len(test_tweets)

# In[825]:

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"
model_size = 300
simple_models , models_by_name = {}, {} 
for view in [v1, v2]:
    simple_models[view] = [
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=model_size, window=3, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DBOW 
        Doc2Vec(dm=0, size=model_size, negative=5, hs=0, min_count=5, workers=cores),
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, size=model_size, window=3, negative=5, hs=0, min_count=2, workers=cores),
    ]

    # speed setup by sharing results of 1st model's vocabulary scan
    simple_models[view][0].build_vocab(alldocs[view])  # PV-DM/concat requires one special NULL word so it serves as template
    print view, simple_models[view][0]
    for model in simple_models[view][1:]:
        model.reset_from(simple_models[view][0])
        print view, model

    models_by_name[view] = OrderedDict((str(model), model) for model in simple_models[view])


# In[826]:

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
for view in [v1, v2]:
    models_by_name[view]['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[view][1], simple_models[view][2]])
    models_by_name[view]['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[view][1], simple_models[view][0]])
#print models_by_name['dbow+dmm'], models_by_name['dbow+dmc'] 
#del models_by_name['dbow+dmc']


# In[827]:

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

def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    predictor = predictor_alg(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
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


# In[828]:

from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
def center(data):
    return data - np.mean(data, axis=0)

def PLS(X, Y):
    cross_cov = np.dot(center(X), center(Y).T)
    eigval,eigvec=np.linalg.eig(cross_cov.dot(cross_cov.T))
    return (eigval, eigvec)

def PLS_MFCC():
    dims = [10, 30, 50, 70, 90, 110]
    accuracies = np.zeros((len(num_neighb), len(dims)))
    # (score, dim, k, PLS_subspace, classifier_object)
    best = (0, 0, 0, None, None) 
    #run pls
    eigval, U = PLS(acoustic_train, artic_train)
    for j, k in enumerate(num_neighb):
        for i, d in enumerate(dims):
            U_d = get_top_eigvec(eigval, U, d)
            #get projection to pls space
            train_proj = np.dot(U_d.T, acoustic_train_cen)
            dev_proj = np.dot(U_d.T, acoustic_dev_cen)
            # stack with mfcc39
            stacked_train = np.append(train_proj, mfcc39_train, axis=0)
            stacked_dev = np.append(dev_proj, mfcc39_dev, axis=0)
            
            #classify
            clf = neighbors.KNeighborsClassifier(k)
            clf.fit(stacked_train.T, phones_train)

            #predictions
            score = clf.score(stacked_dev.T, phones_dev)
            if score > best[0]:
                best = (score, d, k, U_d, clf)
            accuracies[j,i] = score
    return (best, accuracies)

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

def CCA_MFCC():
    dims = [10, 30, 50, 70, 90, 110]
    reg = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e1]
    accuracies = np.zeros((len(reg), len(reg), len(dims), len(num_neighb)))
    # (score, dim, regX, regY, k, CCA_subspace, classifier_object)
    best = (0, 0, 0, 0, 0, None, None)
    for rx, regX in enumerate(reg):
        for ry, regY in enumerate(reg):              
            #run cca
            eigval, U = CCA(acoustic_train, artic_train, regX, regY)
            for i, d in enumerate(dims):
                U_d = get_top_eigvec(eigval, U, d)
                #get projection to cca space
                train_proj = U_d.T.dot(acoustic_train_cen)
                dev_proj = U_d.T.dot(acoustic_dev_cen)
                # stack with mfcc39
                stacked_train = np.append(train_proj, mfcc39_train, axis=0)
                stacked_dev = np.append(dev_proj, mfcc39_dev, axis=0)
                #classify
                for j, k in enumerate(num_neighb):
                    clf = neighbors.KNeighborsClassifier(k)
                    clf.fit(stacked_train.T, phones_train)

                    #predictions
                    score = clf.score(stacked_dev.T, phones_dev)
                    if score > best[0]:
                        best = (score, d, regX, regY, k, U_d, clf)
                    accuracies[rx, ry, i, j] = score
    return (best, accuracies)

def plot_pc2(data, eigvec, phones_data):
    #project to top 2 princ. comp.
    data_proj = np.dot(np.transpose(eigvec), data)
    data_proj_labels=[data_proj[:,np.where(phones_data==lbl)] for lbl in labels_dict.values()]
    #Plot
    cmap = plt.get_cmap('jet_r')
    N=len(labels)
    colors = [cmap(float(i)/N) for i in np.linspace(5.0, 0, N)]
    plt.figure(figsize=(7,7))
    #plt.subplot(2,1,1)
    for i in range(N):
        plt.scatter(data_proj_labels[i][0,:], data_proj_labels[i][1,:] ,c=colors[i], marker='+', label=labels[i]);
    #plt.legend(plots,labels)
    plt.legend(loc=3)
    #plt.show()
    return plt


# In[829]:

#from collections import defaultdict
best_error = dd(lambda: dd(lambda :(1.0, 0.0))) # { view: { model_name : (error_rate, alpha) } } ,to selectively-print only best errors achieved


# In[830]:

predictor_alg = logistic_predictor
#predictor_alg = svm_predictor
from random import shuffle
import datetime

print 'Started.'
for view in [v1, v2]:
    alpha, min_alpha, passes = (0.025, 0.001, 10)
    alpha_delta = (alpha - min_alpha) / passes

    print "======== %s =========" %view
    print("START %s" % datetime.datetime.now())

    for epoch in range(passes):
        shuffle(doc_list[view])  # shuffling gets best results

        for name, train_model in models_by_name[view].items():
            #print name
            # train
            duration = 'na'
            train_model.alpha, train_model.min_alpha = alpha, alpha
            with elapsed_timer() as elapsed:
                train_model.train(doc_list[view])
                duration = '%.1f' % elapsed()

            # evaluate
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs[view], dev_docs[view])
            eval_duration = '%.1f' % eval_elapsed()
            best_indicator = ' '
            if err < best_error[view][name][0]:
                best_error[view][name] = (err, alpha)
                best_indicator = '*' 
            #print("%s%f : %i passes : %s-%s %ss %ss" % (best_indicator, err, epoch + 1, view, name, duration, eval_duration))

            """if ((epoch + 1) % 5) == 0 or epoch == 0:
                eval_duration = ''
                with elapsed_timer() as eval_elapsed:
                    infer_err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs[view], dev_docs[view], infer=True)
                eval_duration = '%.1f' % eval_elapsed()
                best_indicator = ' '
                if infer_err < best_error[view][name + '_inferred'][0]:
                    best_error[view][name + '_inferred'] = (infer_err, alpha)
                    best_indicator = '*'
                print("%s%f : %i passes : %s-%s %ss %ss" % (best_indicator, infer_err, epoch + 1, view, name + '_inferred', duration, eval_duration))
"""
        #print('completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta

    print("END %s" % str(datetime.datetime.now()))


# In[831]:

for view in [v1, v2]:
    print '========= %s ========' %view
    for rate, alpha, name in sorted((rate, alpha, name) for name, (rate, alpha) in best_error[view].items()):
        print("%f %s %f" % (rate, name, alpha))


# In[832]:

doc_id = np.random.randint(simple_models[v1][0].docvecs.count)  # pick random doc; re-run cell for more examples
print('for doc %d...' % doc_id)
# Print example tweet and vector reps for both views
print alldocs['view1'][doc_id]
#tag = alldocs['view1'][doc_id].tags[0]
#print '\n', simple_models['view1'][0].docvecs[tag]

print '\n', alldocs['view2'][doc_id]
#print '\n', simple_models['view2'][0].docvecs[tag]
#print '\n\n', doc_list['view2'][:10]
for model in simple_models[v1]:
    inferred_docvec = model.infer_vector(alldocs[v1][doc_id].words)
    print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))


# In[833]:

#Select the best performing word2vec model
_, best_alpha, best_model_name = min(((rate, alpha, name)                                            for name, (rate, alpha) in best_error[v1].items()), key=lambda b: b[0])
print best_model_name 
print best_alpha
best_model = { v1 : models_by_name[v1][best_model_name],
              v2 : models_by_name[v2][best_model_name] }
# Train best model
shuffle(doc_list[view])
for view in [v1, v2]:
    best_model[view].alpha, best_model[view].min_alpha = best_alpha, best_alpha
    best_model[view].train(doc_list[view])


# In[834]:

# DO CCA on the training docvecs
# X = view 1, Y = view 2 : [word_vec_size x num_samples]
target_sentiments, X, Y = zip(*[(doc.sentiment, best_model[v1].docvecs[doc.tags[0]],                             best_model[v2].docvecs[doc.tags[0]]) for doc in train_docs[v1]])
X = np.asarray(X).T
Y = np.asarray(Y).T
#test docs - view1
dev = [best_model[v1].docvecs[doc.tags[0]] for doc in dev_docs[v1]]
dev = np.asarray(test).T


# In[839]:

(cca_eigval, cca_eigvec) = CCA(X, Y)
#print np.shape(X), np.shape(Y)
#print np.transpose(X)


# In[836]:

predictor = predictor_alg(target_sentiments, X.T)

# predict & evaluate
test_predictions = predictor.predict(dev.T)
predicted = np.rint(dev_predictions)
expected = [doc.sentiment for doc in dev_docs[v1]]
#print("Classification report for %s:\n%s\n" % (predictor, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#ipdb.set_trace()
errors = len(test_predictions) - sum(expected == predicted)
err_orig = float(errors) / len(expected)
print err_orig


# In[837]:

#get top k eigvec, project training data and stack with original word vectors
err_cca = []
step = 5
num_dir_ranges = range(step, step+200, step)
for num_dir in num_dir_ranges:
    top_k_eigv = get_top_eigvec(cca_eigval, cca_eigvec, num_dir)
    #print np.shape(top_k_eigv)
    X_proj = top_k_eigv.T.dot(X)
    #print np.shape(X_proj)
    stacked_vec = np.append(X, X_proj, axis=0)
    #print np.shape(stacked_vec)

    # project dev data to cca directions and stack
    #print np.shape(test)
    test_proj = top_k_eigv.T.dot(dev)
    stacked_dev = np.append(dev, dev_proj, axis=0)
    #print np.shape(stacked_dev)
    #print np.shape(target_sentiments)

    predictor = predictor_alg(target_sentiments, stacked_vec.T)

    # predict & evaluate
    dev_predictions = predictor.predict(stacked_dev.T)
    predicted = np.rint(dev_predictions)
    expected = [doc.sentiment for doc in dev_docs[v1]]
    #print("Classification report for %s:\n%s\n" % (predictor, metrics.classification_report(expected, predicted)))
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    #ipdb.set_trace()
    errors = len(test_predictions) - sum(expected == predicted)
    err_cca.append(float(errors) / len(expected))


# In[838]:

plt.plot(num_dir_ranges, [err_orig]*len(num_dir_ranges), 'r--', num_dir_ranges, err_cca, 'b*')
plt.ylabel('Mis-classification rate')
plt.xlabel('Number of CCA directions stacked with original sentence vectors')
plt.grid()
#plt.axis([0, 200, 0.35, 0.4])
plt.show()


# In[ ]:



