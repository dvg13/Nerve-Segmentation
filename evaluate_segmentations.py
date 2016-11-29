import numpy as np
from scipy import misc
from sklearn.ensemble import RandomForestClassifier
import torchfile
import math
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

#train_directory
train_directory = "/home/dvgainer/NERVE/DATA/train/"

#raw_directory
raw_directory = "/home/dvgainer/NERVE/TEMP/segmentations/35_new_double_aug_train/"

#clean_directory
clean_directory = "/home/dvgainer/NERVE/TEMP/segmentations/35_new_double_aug_train_clean/"

#get the embeddings and targets - these are both numerically indexed
embeddings = torchfile.load("/home/dvgainer/NERVE/TEMP/embeddings/35_new_double_train.torch")

#
train_size = 3957
total_size = 5635

targets = torchfile.load("/home/dvgainer/NERVE/TEMP/class_targets.torch")
embeddings.resize(5635,23*30)


#get the two sets of classification images - these are indexed with number and name
image_list_file = ("/home/dvgainer/NERVE/TEMP/image_list")
image_list_fh = open(image_list_file)
image_name_idx = {}
idx = 0
for line in image_list_fh:
	image_name_idx[idx] = line.strip()
	idx += 1

#dice function
def dice(image1,image2):
	if np.sum(image1) == 0 and np.sum(image2) == 0:
		return 1
	else:
		return 2 * np.sum(image1 * image2) / (np.sum(image1) + np.sum(image2))

#imread functions
def get_raw(i):
	return misc.imread(raw_directory + image_name_idx[i] + ".png") / 255
def get_clean(i):
	return misc.imread(clean_directory + image_name_idx[i] + ".png") / 255
def get_true(i):
	return misc.imread(train_directory + image_name_idx[i] + "_mask.tif") / 255

#get_dice_score_from_idx
def get_dice(i,seg_type="CLEAN"):
	true_mask = get_true(i)
	
	if seg_type == "RAW": 
		raw_mask = get_raw(i)
		return dice(true_mask,raw_mask)
	elif seg_type == "CLEAN":
		clean_mask = get_clean(i)
		return dice(true_mask,clean_mask)
	elif seg_type == "BOTH":
		raw_mask = get_raw(i)
		clean_mask = get_clean(i)
		return (dice(true_mask,raw_mask),dice(true_mask,clean_mask))

#simple threhsolding of the maximum pixel intensity in the image
def get_max_classifier(idx):
	pos_values = np.zeros(len(idx))
	neg_values = np.zeros_like(pos_values)

	i_p = 0
	i_n = 0	

	for i in idx:
		if targets[i] == 1:
			pos_values[i_p] = np.max(get_raw(i) * get_clean(i))
			i_p += 1
		else:
			neg_values[i_n] = np.max(get_raw(i) * get_clean(i))
			i_n += 1
		
	pos_values.resize(i_p)
	neg_values.resize(i_n)

	
	pos_mean = np.mean(pos_values)
	pos_std = np.std(pos_values)

	neg_mean = np.mean(neg_values)
	neg_std = np.std(neg_values)

	print(pos_mean,pos_std,neg_mean,neg_std)

#simple threhsolding of the total pixel intensity in the image
def test_max_classifier(idx,m_pos,std_pos,m_neg,std_neg):

	dice_total = 0
	correct = 0
	fp = 0
	fn = 0 
	tp = 0
	tn = 0

	
	for i in idx:
		x = np.max(get_raw(i) * get_clean(i))

		pos = stats.norm(m_pos,std_pos).pdf(x)
		neg = stats.norm(m_neg,std_neg).pdf(x)

		if pos > neg:
			if targets[i] == 1:
				correct += 1
				tp += 1
				dice_total += dice(get_clean(i),get_true(i))
			else:
				fp += 1
		else:
			if targets[i] == 0:
				correct += 1
				tn += 1
				dice_total += 1
			else:
				fn +=1

	print(correct / len(idx), dice_total / len(idx), tp,tn,fp,fn)

def get_total_classifier(idx):
	pos_values = np.zeros(len(idx))
	neg_values = np.zeros_like(pos_values)

	i_p = 0
	i_n = 0	

	for i in idx:
		if targets[i] == 1:
			pos_values[i_p] = np.sum(get_raw(i) * get_clean(i))
			i_p += 1
		else:
			neg_values[i_n] = np.sum(get_raw(i)  * get_clean(i))
			i_n += 1
		
	pos_values.resize(i_p)
	neg_values.resize(i_n)

	
	pos_mean = np.mean(pos_values)
	pos_std = np.std(pos_values)

	neg_mean = np.mean(neg_values)
	neg_std = np.std(neg_values)

	print(pos_mean,pos_std,neg_mean,neg_std)

def test_total_classifier(idx,m_pos,std_pos,m_neg,std_neg):

	dice_total = 0
	correct = 0
	fp = 0
	fn = 0 
	tp = 0
	tn = 0

	
	for i in idx:
		x = np.sum(get_raw(i) * get_clean(i))
		#print(x)

		pos = stats.norm(m_pos,std_pos).pdf(x)
		neg = stats.norm(m_neg,std_neg).pdf(x)

		#print(pos,neg)
		#pause = input()

		if pos > neg:
			if targets[i] == 1:
				correct += 1
				tp += 1
				dice_total += dice(get_clean(i),get_true(i))
			else:
				fp += 1
		else:
			if targets[i] == 0:
				correct += 1
				tn += 1
				dice_total += 1
			else:
				fn +=1

	print(correct / len(idx), dice_total / len(idx), tp,tn,fp,fn)

def test_forest(idx,classifier,refine=float('inf')):

	correct = 0
	dice_total = 0
	fp = 0
	fn = 0 
	tp = 0
	tn = 0

	
	for i in idx:
		y = classifier.predict(embeddings[i].reshape(1,-1))[0]

		#here we can flip the result if the stats from the image
		#signal strongly in favor of doing so
		if refine < float('inf'):
			total = np.sum(get_raw(i) * get_clean(i))
			pos = stats.norm(3170.72964217, 1427.35785348).pdf(total)
			neg = stats.norm(867.501292451, 957.592899154).pdf(total)
			
			if y == 0 and pos/neg > refine:
				y = 1
			if y == 1 and neg/pos > refine:
				y = 0

			 


		if y == 1:
			if targets[i] == 1:
				correct += 1
				tp += 1
				dice_total += dice(get_clean(i),get_true(i))
			else:
				fp += 1

		else:
			if targets[i] == 0:
				correct += 1
				tn += 1
				dice_total += 1
			else:
				fn +=1

	print(correct / len(idx), dice_total / len(idx),tp,tn,fp,fn)
	
def train_forest(idx,n):
	classifier = RandomForestClassifier(n,class_weight="balanced")
	classifier.fit(embeddings[idx],targets[idx])
	print(classifier.score(embeddings[idx],targets[idx]))

	return classifier



