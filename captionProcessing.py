#1
# Import all required modules
import nltk

# DO this download only once
nltk.download('stopwords')
nltk.download('punkt')

import itertools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from PIL import Image
from PIL import ImageFilter
import os
import time



# generate all stop words
stop_words = set(stopwords.words("english"))
# print(f"Stop Words are:- {stop_words} ")



# load caption file and separate each image_id along with image description

def load_caption_file(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text
 
# extract image descriptions
def load_image_descriptions(doc):
	mapping = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		if len(line) < 2:
			continue
		image_id, image_desc = tokens[0], tokens[1:]
		image_id = image_id.split('.')[0]
		image_desc = ' '.join(image_desc)
		if image_id not in mapping:
			mapping[image_id] = image_desc
	return mapping
 

# path of caption/token/description file for dataset and all function calls
filename = r'/content/drive/MyDrive/flickr/Flickr8k.token.txt'
doc = load_caption_file(filename)
descriptions = load_image_descriptions(doc)
print("For image id 1007320043_627395c3d8 :-", descriptions['1007320043_627395c3d8'])
print('Loaded: %d ' % len(descriptions))



# for printing only first five descriptions from dictionary
# x = itertools.islice(descriptions.items(), 0, 5)



#2 (for test data)
# remove stopwords and words having length < 2.....store all valid words in contain_all_words,a set() data str.
# Separate dictionaries are created having id and list of words in that image id.....finally added to the list->img_list
def create_token(img_id,img_cap):
	wrd = word_tokenize(img_cap)
	res = []
	for w in wrd:
		if w not in stop_words and len(w) > 2:
			contain_all_words.append(w)
			res.append(w)
	d = {img_id : res}
	img_list.append(d)



# obtain id and desc from dictionary of filtered images
def remove_stopwords(my_dict,descriptions):
	for img_id in my_dict.keys():
		if img_id not in descriptions: 
			continue
		create_token(img_id,descriptions[img_id])



def filter_images():
	inPath ="drive/MyDrive/flickr/Images_test"
	my_dict = dict()
	print("Running.....")
	c = 0
	for imagePath in os.listdir(inPath):
		# imagePath contains name/id of the image
		inputPath = os.path.join(inPath, imagePath)
		img = Image.open(inputPath)
		imagePath = imagePath.split('.')[0]
		my_dict[imagePath]=1
		c += 1

	print(f"Total Entries of images :- {c}")
	return my_dict


contain_all_words = list()
img_list = list()

my_dict = filter_images()
remove_stopwords(my_dict,descriptions)
total_ids = len(img_list)
total_words = len(contain_all_words)

print(f"Image Dataset Description:- {total_ids}")
print(f"A list containg all the words : - {total_words}")


#3 (for train data)
# remove stopwords and words having length < 3.....store all valid words in contain_all_words,a set() data str.
# Separate dictionaries are created having id and list of words in that image id.....finally added to the list->img_list
def create_token1(img_id,img_cap):
	wrd = word_tokenize(img_cap)
	res = []
	for w in wrd:
		if w not in stop_words and len(w) > 2:
			contain_all_words.append(w)
			res.append(w)
	d = {img_id : res}
	img_list1.append(d)



# obtain id and desc from dictionary of filtered images
def remove_stopwords1(my_dict,descriptions):
	for img_id in my_dict.keys():
		if img_id not in descriptions: 
			continue
		create_token1(img_id,descriptions[img_id])



def filter1():
	inPath ="drive/MyDrive/flickr/Images_train"
	my_dict = dict()
	print("Running.....")
	c = 0
	for imagePath in os.listdir(inPath):
		# imagePath contains name/id of the image
		inputPath = os.path.join(inPath, imagePath)
		img = Image.open(inputPath)
		#if ((img.size[0]*img.size[1])==166500) and (img.size[0]==500):
		imagePath = imagePath.split('.')[0]
		my_dict[imagePath]=1
		c += 1

	print(f"Total Entries of images :- {c}")
	return my_dict


#contain_all_words1 = list()
img_list1 = list()

my_dict1 = filter1()
remove_stopwords1(my_dict1,descriptions)
total_ids = len(img_list1)
total_words = len(contain_all_words)

print(f"Image Dataset Description:- {total_ids}")
print(f"A list containg all the words : - {total_words}")

# 4
#Creating a list of all unique words with frequency>1

fr1=dict()
for word in contain_all_words:
    if(word in fr1):
        fr1[word]+=1
    else:
        fr1[word]=1

unique_words1=[]
for x,y in fr1.items():
    if y>1:
        unique_words1.append(x)
        
print(f"Total unique words in both test and train data are:-{len(unique_words1)}")

#5
#using POS tagger 
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize

tag=[]
for i in unique_words1:
      
    wordsList = nltk.word_tokenize(i)
  
    tagged = nltk.pos_tag(wordsList)
    tag.append(tagged)

#6
noun=[] #containing singular nouns
i=0
for x in tag:

    if x[0][1] =='NN':
        noun.append(x[0][0])

total_unique_words1=len(noun)
print(f"total unique singular nouns occurring atleast twice: {total_unique_words1}")

#7
# Create matrix using dictionary for test data
start_time = time.time()

mat = dict()
# l = [0]*total_unique_words#earlier total_words was used

for d in img_list:
	k = list(d.keys())
	v = d[k[0]]
	v = set(v)
	k = k[0]
	mat[k] = [0]*total_unique_words1
	i = 0
	for w in noun:#earlier contain_all_words was used
		if w in v:
			mat[k][i] = 1#(1,w)
		else:
			mat[k][i] = 0
		i += 1



# For printing entire matrix...:o :o :o :(

print(f"\n for id 1007320043_627395c3d8 - {mat['1007320043_627395c3d8']}")
# for k,v in mat.items():
# 	print(k,v)


end_time = time.time()
print("Time Taken for creation of matrix :-",abs(end_time-start_time))

#8
#creating 2D numpy array (test labels) from above dictionary
import numpy as np
l=list()

for x,y in mat.items():
    l.append(y)

y_test=np.array(l)
print(y_test.shape)

#9
# Create matrix using dictionary for training data
start_time1 = time.time()

mat1 = dict()
# l = [0]*total_unique_words#earlier total_words was used

for d in img_list1:
	k = list(d.keys())
	v = d[k[0]]
	v = set(v)
	k = k[0]
	mat1[k] = [0]*total_unique_words1
	i = 0
	for w in noun:#earlier contain_all_words was used
		if w in v:
			mat1[k][i] = 1#(1,w)
		else:
			mat1[k][i] = 0
		i += 1



# For printing entire matrix...:o :o :o :(

print(f"\n for id 1417295167_5299df6db8 - {mat1['1417295167_5299df6db8']}")
# for k,v in mat.items():
# 	print(k,v)


end_time1 = time.time()
print("Time Taken for creation of matrix :-",abs(end_time1-start_time1))

#10
#creating 2D numpy array (train labels) from above dictionary
l=list()

for x,y in mat1.items():
    l.append(y)

y_train=np.array(l)

print(y_train.shape)

