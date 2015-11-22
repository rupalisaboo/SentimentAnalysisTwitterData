import os, shutil

input_file = raw_input('Input csv file [./corpus.csv]: ')
batch_dir_pref = raw_input('Batch dir prefix [./batch_]: ')
raw_dir = raw_input('raw data dir [./rawdata/]: ')


if not input_file:
	input_file = './corpus.csv'
if not batch_dir_pref:
	batch_dir_pref = './batch'
if not raw_dir:
	input_file = './rawdata/'
elif not raw_dir.endswith('/'):
	raw_dir = raw_dir + '/'

num_batches = 4
ids = []
with open(raw_input, 'w') as fh:
	for line in fh.readlines():
		[_,_,id] = line.strip().split(',')
		ids.append(id.strip())
total = len(ids)
batch_sz = total/num_batches
batches = [ids[i * batch_sz : (i+1) * batch_sz] for i in range(num_batches)]
if total > batch_sz*num_batches:
	batches[0].append(ids[batch_sz*num_batches:])
for i in range(num_batches):
	batch_i_dir = '%s_%s/' %(batch_dir_pref,i)
	if not os.path.exists(batch_i_dir):
		os.mkdir(batch_i_dir)
	for tweet_id in batches[i]:
		tweet_file = '%s%s.json' %(raw_dir, tweet_id)
		if not os.path.exists(tweet_file):
			print 'Skipping %s: File not found.' %(tweet_file)
		shutil.copyfile(tweet_file, batch_i_dir)
