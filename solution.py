import tensorflow as tf 
import numpy as np 
import csv
import os
from neural_network import neural_network

with open(os.path.join(os.getcwd(),'data','test.csv'),'r') as csvfile:
	csvreader = csv.reader(csvfile)
	csvreader.next()
	test_data = []
	for row in csvreader:
		row = [int(i) for i in row]
		test_data.append(row)

test_data = np.asarray(test_data)

network = neural_network.neural_network()

BATCH_SIZE = 1
is_training = False
network.create_model()
sess = tf.Session()
network.session_init(sess)
network.load_weights('log_data/2018-10-06-04-29-41/weights/90.ckpt')

with open(os.path.join(os.getcwd(),'data','submission.csv'),'w') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(['ImageId','Label'])

def log_data(data):
	with open(os.path.join(os.getcwd(),'data','submission.csv'),'a') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(data)	

for idx in range(test_data.shape[0]):
# for idx in range(10):
	current_data = test_data[idx,:].reshape((1,784))
	pred = network.forward(current_data,is_training,BATCH_SIZE)
	pred_value = np.argmax(pred)
	log_data([idx+1,pred_value])
