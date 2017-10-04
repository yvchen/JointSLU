"""
RNN-based slot filling
by V. Chen, D. Hakkani-Tur & G. Tur
"""

import os, sys, json
import numpy as np 
from scipy import io
from wordSlotDataSet import dataSet, readNum
from PredefinedEmbedding import PredefinedEmbedding
from Encoding import encoding
import argparse
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, merge, Merge, Dense, Dropout, Activation, RepeatVector, Permute, Reshape, RepeatVector, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax
from keras.constraints import maxnorm, nonneg
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from History import LossHistory

class KerasModel( object ):

	def __init__(self,argparams):
		# PARAMETERS
		self.hidden_size = argparams['hidden_size'] # size of hidden layer of neurons 
		self.learning_rate = argparams['learning_rate']
		self.training_file = argparams['train_data_path']
		self.validation_file = argparams['dev_data_path']
		self.test_file = argparams['test_data_path']
		self.result_path = argparams['result_path']
		self.train_numfile = argparams['train_numfile']
		self.dev_numfile = argparams['dev_numfile']
		self.test_numfile = argparams['test_numfile']
		self.update_f = argparams['sgdtype'] # options: adagrad, rmsprop, vanilla. default: vanilla
		self.decay_rate = argparams['decay_rate'] # for rmsprop
		self.default = argparams['default_flag'] # True: use defult values for optimizer
		self.momentum = argparams['momentum'] # for vanilla update
		self.max_epochs = argparams['max_epochs']
		self.activation = argparams['activation_func'] # options: tanh, sigmoid, relu. default: relu
		self.smooth_eps = argparams['smooth_eps'] # for adagrad and rmsprop
		self.batch_size = argparams['batch_size']
		self.input_type = argparams['input_type'] # options: 1hot, embedding, predefined
		self.emb_dict = argparams['embedding_file']
		self.embedding_size = argparams['embedding_size']
		self.dropout = argparams['dropout']
		self.dropout_ratio = argparams['dropout_ratio']
		self.iter_per_epoch = argparams['iter_per_epoch']
		self.arch = argparams['arch']
		self.init_type = argparams['init_type']
		self.fancy_forget_bias_init = argparams['forget_bias']
		self.time_length = argparams['time_length']
		self.his_length = argparams['his_length']
		self.mdl_path = argparams['mdl_path']
		self.log = argparams['log']
		self.record_epoch = argparams['record_epoch']
		self.load_weight = argparams['load_weight']
		self.combine_his = argparams['combine_his']
		self.time_decay = argparams['time_decay']
		self.shuffle = argparams['shuffle']
		self.set_batch = argparams['set_batch']
		self.tag_format = argparams['tag_format']
		self.e2e_flag = argparams['e2e_flag']
		self.output_att = argparams['output_att']
		self.sembedding_size = self.embedding_size
		self.model_arch = self.arch
		if self.validation_file is None:
			self.nodev = True
		else:
			self.nodev = False
		if self.input_type == 'embedding':
			self.model_arch = self.model_arch + '+emb'
		if self.time_decay:
			self.model_arch = self.model_arch + '+T'
		if self.e2e_flag:
			self.model_arch = 'e2e-' + self.model_arch

	def test(self, H, X, data_type, tagDict, pad_data):
		# open a dir to store results
		if self.default:
			target_file = self.result_path + '/' + self.model_arch + '_H-'+str(self.hidden_size)+'_O-'+self.update_f+'_A-'+self.activation+'_WR-'+self.input_type
		else:
			target_file = self.result_path + '/' + self.model_arch +'-LR-'+str(self.learning_rate)+'_H-'+str(self.hidden_size)+'_O-'+self.update_f+'_A-'+self.activation+'_WR-'+self.input_type

		if 'memn2n' in self.arch or self.arch[0] == 'h':
			batch_data = [H, X]
		else:
			batch_data = X

		# output attention
		if self.output_att is not None:
			x1 = self.model.inputs[0]
			x2 = self.model.inputs[1]
			#x = self.model.layers[1].input
			y = self.model.get_layer(name='match').output
#			y = self.model.layers[9].output
			f = K.function([x1, x2, K.learning_phase()], y)
			att_mtx = f([batch_data[0], batch_data[1], 0])
			row, col = np.shape(att_mtx)
			fo = open(self.output_att, 'wb')
			for i in range(0, row):
				for j in range(0, col):
					fo.write("%e " %att_mtx[i][j])
				fo.write('\n')
			fo.close()
			sys.stderr.write("Output the attention weights in the file %s.\n" %self.output_att)
			exit()
		if "predict_classes" in dir(self.model):
			prediction = self.model.predict_classes(batch_data)
			probability = self.model.predict_proba(batch_data)
		else:
			probability = self.model.predict(batch_data)
			prediction = np.argmax(probability, axis=2)

		# output prediction and probability results
		fo = open(target_file+"."+ data_type, "wb")
		for i, sent in enumerate(prediction):
			for j, tid in enumerate(sent):
				if pad_data[i][j] != 0:
					if self.tag_format == 'normal':
						fo.write(tagDict[tid] + ' ')
					elif self.tag_format == 'conlleval':
						fo.write(tagDict[tid] + '\n')
			fo.write('\n')
		fo.close()
		fo = open(target_file+"."+ data_type+'.prob', "wb")
		for i, sent in enumerate(probability):
			for j, prob in enumerate(sent):
				if pad_data[i][j] != 0:
					for k, val in enumerate(prob):
						fo.write("%e " %val)
					fo.write("\n")
		fo.close()

	def build( self ):

		# decide main model
		if self.input_type == '1hot':
			self.embedding_size = self.input_vocab_size

		# set optimizer
		opt_func = self.update_f
		if not self.default:
			if self.update_f == 'sgd':
				opt_func = SGD(lr=self.learning_rate, momentum=self.momentum, decay=self.decay_rate)
			elif self.update_f == 'rmsprop':
				opt_func = RMSprop(lr=self.learning_rate, rho=self.rho, epsilon=self.smooth_eps)
			elif self.update_f == 'adagrad':
				opt_func = Adagrad(lr=self.learning_rate, epsilon=self.smooth_eps)
			elif self.update_f == 'adadelta':
				opt_func = Adadelta(lr=self.learning_rate, rho=self.rho, epsilon=self.smooth_eps)
			elif self.update_f == 'adam':
				opt_func = Adam(lr=self.learning_rate, beta_1=self.beta1, beta_2=self.beta2, epsilon=self.smooth_eps)
			elif self.update_f == 'adamax':
				opt_func = Adamax(lr=self.learning_rate, beta_1=self.beta1, beta_2=self.beta2, epsilon=self.smooth_eps)
			else:
				sys.stderr.write("Invalid optimizer.\n")
				exit()

		# Vallina RNN (LSTM, SimpleRNN, GRU)
		# Bidirectional-RNN (LSTM, SimpleRNN, GRU)
		if self.arch == 'lstm' or self.arch == 'rnn' or self.arch == 'gru' or self.arch == 'blstm' or self.arch == 'brnn' or self.arch == 'bgru':
			if self.input_type == 'embedding':
				raw_current = Input(shape=(self.time_length,), dtype='int32')
				current = Embedding(input_dim=self.input_vocab_size, output_dim=self.embedding_size, input_length=self.time_length, mask_zero=True)(raw_current)
			else:
				current = raw_current = Input(shape=(self.time_length, self.input_vocab_size))
			if 'rnn' in self.arch:
				forward = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				backward = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(current)
			elif 'gru' in self.arch:
				forward = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				backward = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(current)
			elif 'lstm' in self.arch:
				forward = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				backward = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(current)
			if 'b' in self.arch:
				tagger = merge([forward, backward], mode='concat')
			else:
				tagger = forward
			if self.dropout:
				tagger = Dropout(self.dropout_ratio)(tagger)
			prediction = TimeDistributed(Dense(self.output_vocab_size, activation='softmax'))(tagger)

			self.model = Model(input=raw_current, output=prediction)
			self.model.compile(loss='categorical_crossentropy', optimizer=opt_func)

		# 2-Stacked Layered RNN (LSTM, SimpleRNN, GRU)
		elif self.arch == '2lstm' or self.arch == '2rnn' or self.arch == '2gru':
			self.model = Sequential()
			if self.input_type == 'embedding':
				self.model.add(Embedding(self.input_vocab_size, self.embedding_size, input_length=self.time_length))
			if self.arch == '2lstm':
				basic_model = LSTM(self.hidden_size, return_sequences=True, input_shape=(self.time_length, self.embedding_size), init=self.init_type, activation=self.activation)
				stack_model = LSTM(self.hidden_size, return_sequences=True, input_shape=(self.time_length, self.hidden_size), init=self.init_type, activation=self.activation)
			elif self.arch == '2rnn':
				basic_model = SimpleRNN(self.hidden_size, return_sequences=True, input_shape=(self.time_length, self.embedding_size), init=self.init_type, activation=self.activation)
				stack_model = SimpleRNN(self.hidden_size, return_sequences=True, input_shape=(self.time_length, self.hidden_size), init=self.init_type, activation=self.activation)
			else:
				basic_model = GRU(self.hidden_size, return_sequences=True, input_shape=(self.time_length, self.embedding_size), init=self.init_type, activation=self.activation)	
				stack_model = GRU(self.hidden_size, return_sequences=True, input_shape=(self.time_length, self.hidden_size), init=self.init_type, activation=self.activation)
			self.model.add(basic_model)
			if self.dropout:
				self.model.add(Dropout(self.dropout_ratio))
			self.model.add(stack_model)
			if self.dropout:
				self.model.add(Dropout(self.dropout_ratio))
			self.model.add(TimeDistributed(Dense(self.output_vocab_size)))
			self.model.add(Activation('softmax'))
			self.model.compile(loss='categorical_crossentropy', optimizer=opt_func)

		# Encode intent information by feeding all words and then start tagging
		elif self.arch == 'irnn' or self.arch == 'igru' or self.arch == 'ilstm' or self.arch == 'ibrnn' or self.arch == 'ibgru' or self.arch == 'iblstm':
			if self.input_type == 'embedding':
				raw_current = Input(shape=(self.time_length,), dtype='int32')
				current = Embedding(input_dim=self.input_vocab_size, output_dim=self.embedding_size, input_length=self.time_length)(raw_current)
			else:
				current = raw_current = Input(shape=(self.time_length, self.input_vocab_size))
			if 'rnn' in self.arch:
				fencoder = SimpleRNN(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
				bencoder = SimpleRNN(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation, go_backwards=True)(current)
				flabeling = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				blabeling = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(current)
			elif 'gru' in self.arch:
				fencoder = GRU(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
				bencoder = GRU(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation, go_backwards=True)(current)
				flabeling = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				blabeling = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(current)
			elif 'lstm' in self.arch:
				fencoder = LSTM(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
				bencoder = LSTM(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation, go_backwards=True)(current)
				flabeling = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				blabeling = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(current)
			if 'b' in self.arch:
				encoder = merge([fencoder, bencoder], mode='concat', concat_axis=-1)
				labeling = merge([flabeling, blabeling], mode='concat', concat_axis=-1)
			else:
				encoder = fencoder
				labeling = flabeling
			#intent = Dense(self.output_vocab_size, activation='softmax')(encoder)
			encoder = RepeatVector(self.time_length)(encoder)
			tagger = merge([encoder, labeling], mode='concat', concat_axis=-1)
			if self.dropout:
				tagger = Dropout(self.dropout_ratio)(tagger)
			prediction = TimeDistributed(Dense(self.output_vocab_size, activation='softmax'))(tagger)

			self.model = Model(input=raw_current, output=prediction)
			self.model.compile(loss='categorical_crossentropy', optimizer=opt_func)

		# Encode intent information by feeding all words and then start tagging
		elif self.arch == 'i-c-rnn' or self.arch == 'i-c-gru' or self.arch == 'i-c-lstm' or self.arch == 'i-c-brnn' or self.arch == 'i-c-bgru' or self.arch == 'i-c-blstm':
			if self.input_type == 'embedding':
				raw_current = Input(shape=(self.time_length,), dtype='int32')
				current = Embedding(input_dim=self.input_vocab_size, output_dim=self.embedding_size, input_length=self.time_length)(raw_current)
			else:
				current = raw_current = Input(shape=(self.time_length, self.input_vocab_size))
			encoder = Convolution1D(self.embedding_size, 3, border_mode='same', input_shape=(self.time_length, self.embedding_size))(current)
			encoder = MaxPooling1D(self.time_length)(encoder)
			encoder = Flatten()(encoder)
			if 'rnn' in self.arch:
				forward = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				backward = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(current)
			elif 'gru' in self.arch:
				forward = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				backward = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(current)
			elif 'lstm' in self.arch:
				forward = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				backward = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(current)
			if 'b' in self.arch:
				labeling = merge([forward, backward], mode='concat', concat_axis=-1)
			else:
				labeling = forward
			#intent = Dense(self.output_vocab_size, activation='softmax')(encoder)
			encoder = RepeatVector(self.time_length)(encoder)
			tagger = merge([encoder, labeling], mode='concat', concat_axis=-1)
			if self.dropout:
				tagger = Dropout(self.dropout_ratio)(tagger)
			prediction = TimeDistributed(Dense(self.output_vocab_size, activation='softmax'))(tagger)

			self.model = Model(input=raw_current, output=prediction)
			self.model.compile(loss='categorical_crossentropy', optimizer=opt_func)


		# Encode all history and the current utterance first and then start tagging
		elif self.arch == 'hirnn' or self.arch == 'higru' or self.arch == 'hilstm' or self.arch == 'hibrnn' or self.arch == 'hibgru' or self.arch == 'hiblstm':
			if self.input_type == 'embedding':
				raw_his = Input(shape=(self.time_length * self.his_length,), dtype='int32')
				raw_cur = Input(shape=(self.time_length,), dtype='int32')
				his_vec = Embedding(input_dim=self.input_vocab_size, output_dim=self.embedding_size, input_length=(self.time_length * self.his_length))(raw_his)
				cur_vec = Embedding(input_dim=self.input_vocab_size, output_dim=self.embedding_size, input_length=self.time_length)(raw_cur)
			else:
				his_vec = raw_his = Input(shape=(self.time_length * self.his_length, self.input_vocab_size))
				cur_vec = raw_cur = Input(shape=(self.time_length, self.input_vocab_size))
			if 'rnn' in self.arch:
				fencoder = SimpleRNN(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation)(his_vec)
				bencoder = SimpleRNN(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation, go_backwards=True)(his_vec)
				flabeling = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(cur_vec)
				blabeling = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(cur_vec)
			elif 'gru' in self.arch:
				fencoder = GRU(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation)(his_vec)
				bencoder = GRU(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation, go_backwards=True)(his_vec)
				flabeling = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(cur_vec)
				blabeling = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(cur_vec)
			elif 'lstm' in self.arch:
				fencoder = LSTM(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation)(his_vec)
				bencoder = LSTM(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation, go_backwards=True)(his_vec)
				flabeling = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(cur_vec)
				blabeling = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(cur_vec)
			if 'b' in self.arch:
				encoder = merge([fencoder, bencoder], mode='concat', concat_axis=-1)
				labeling = merge([flabeling, blabeling], mode='concat', concat_axis=-1)
			else:
				encoder = fencoder
				labeling = flabeling
			#intent = Dense(self.output_vocab_size, activation='softmax')(encoder)
			encoder = RepeatVector(self.time_length)(encoder)
			tagger = merge([encoder, labeling], mode='concat', concat_axis=-1)
			if self.dropout:
				tagger = Dropout(self.dropout_ratio)(tagger)
			prediction = TimeDistributed(Dense(self.output_vocab_size, activation='softmax'))(tagger)

			self.model = Model(input=[raw_his, raw_cur], output=prediction)
			self.model.compile(loss='categorical_crossentropy', optimizer=opt_func)

		# Encode all history and the current utterance first and then start tagging
		elif self.arch == 'hi-c-rnn' or self.arch == 'hi-c-gru' or self.arch == 'hi-c-lstm' or self.arch == 'hi-c-brnn' or self.arch == 'hi-c-bgru' or self.arch == 'hi-c-blstm':
			if self.input_type == 'embedding':
				raw_his = Input(shape=(self.time_length * self.his_length,), dtype='int32')
				raw_cur = Input(shape=(self.time_length,), dtype='int32')
				his_vec = Embedding(input_dim=self.input_vocab_size, output_dim=self.embedding_size, input_length=(self.time_length * self.his_length))(raw_his)
				cur_vec = Embedding(input_dim=self.input_vocab_size, output_dim=self.embedding_size, input_length=self.time_length)(raw_cur)
			else:
				his_vec = raw_his = Input(shape=(self.time_length * self.his_length, self.input_vocab_size))
				cur_vec = raw_cur = Input(shape=(self.time_length, self.input_vocab_size))
			if 'rnn' in self.arch:
				flabeling = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(cur_vec)
				blabeling = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(cur_vec)
			elif 'gru' in self.arch:
				flabeling = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(cur_vec)
				blabeling = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(cur_vec)
			elif 'lstm' in self.arch:
				flabeling = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(cur_vec)
				blabeling = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(cur_vec)
			if 'b' in self.arch:
				labeling = merge([flabeling, blabeling], mode='concat', concat_axis=-1)
			else:
				labeling = flabeling
			encoder = Convolution1D(self.embedding_size, 3, border_mode='same', input_shape=(self.time_length * self.his_length, self.embedding_size))(his_vec)
			encoder = MaxPooling1D(self.time_length)(encoder)
			encoder = Flatten()(encoder)
			#intent = Dense(self.output_vocab_size, activation='softmax')(encoder)
			encoder = RepeatVector(self.time_length)(encoder)
			tagger = merge([encoder, labeling], mode='concat', concat_axis=-1)
			if self.dropout:
				tagger = Dropout(self.dropout_ratio)(tagger)
			prediction = TimeDistributed(Dense(self.output_vocab_size, activation='softmax'))(tagger)

			self.model = Model(input=[raw_his, raw_cur], output=prediction)
			self.model.compile(loss='categorical_crossentropy', optimizer=opt_func)

		elif 'amemn2n' in self.arch:
			# current: (, time_length, embedding_size)
			if self.input_type == 'embedding':
				raw_current = Input(shape=(self.time_length,), dtype='int32', name='raw_current')
				current = Embedding(input_dim=self.input_vocab_size, output_dim=self.embedding_size, input_length=self.time_length)(raw_current)
			else:
				current = raw_current = Input(shape=(self.time_length, self.input_vocab_size), name='current')
			if 'memn2n-d-' in self.arch:
				cur_vec = TimeDistributed(Dense(self.sembedding_size, input_shape=(self.time_length, self.embedding_size)))(current)
				cur_vec = AveragePooling1D(self.time_length)(cur_vec)
				cur_vec = Flatten()(cur_vec)
			elif 'memn2n-c-' in self.arch:
				cur_vec = Convolution1D(self.sembedding_size, 3, border_mode='same', input_shape=(self.time_length, self.embedding_size))(current)
				cur_vec = MaxPooling1D(self.time_length)(cur_vec)
				cur_vec = Flatten()(cur_vec)
			elif 'memn2n-r-' in self.arch:
				if 'blstm' in self.arch:
					fcur_vec = LSTM(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
					bcur_vec = LSTM(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation, go_backwards=True)(current)
					cur_vec = merge([fcur_vec, bcur_vec], mode='concat')
				elif 'rnn' in self.arch:
					cur_vec = SimpleRNN(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
				elif 'gru' in self.arch:
					cur_vec = GRU(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
				elif 'lstm' in self.arch:
					cur_vec = LSTM(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
				else:
					sys.stderr.write("The RNN model is invaliad. (rnn | gru | lstm)\n")
			elif 'memn2n-rc-' in self.arch:
				if 'rnn' in self.arch:
					cur_vec = SimpleRNN(self.sembedding_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				elif 'gru' in self.arch:
					cur_vec = GRU(self.sembedding_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				elif 'lstm' in self.arch:
					cur_vec = LSTM(self.sembedding_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				else:
					sys.stderr.write("The RNN model is invaliad. (rnn | gru | lstm)\n")
				cur_vec = Convolution1D(self.sembedding_size, 3, border_mode='same', input_shape=(self.time_length, self.sembedding_size))(cur_vec)
				cur_vec = MaxPooling1D(self.time_length)(cur_vec)
				cur_vec = Flatten()(cur_vec)
			elif 'memn2n-cr-' in self.arch:
				cur_vec = Convolution1D(self.sembedding_size, 3, border_mode='same', input_shape=(self.time_length, self.embedding_size))(current)
				if 'rnn' in self.arch:
					cur_vec = SimpleRNN(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				elif 'gru' in self.arch:
					cur_vec = GRU(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				elif 'lstm' in self.arch:
					cur_vec = LSTM(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				else:
					sys.stderr.write("The RNN model is invaliad. (rnn | gru | lstm)\n")
			elif 'memn2n-crp-' in self.arch:
				cur_vec = Convolution1D(self.sembedding_size, 3, border_mode='same', input_shape=(self.time_length, self.embedding_size))(current)
				if 'rnn' in self.arch:
					cur_vec = SimpleRNN(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				elif 'gru' in self.arch:
					cur_vec = GRU(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				elif 'lstm' in self.arch:
					cur_vec = LSTM(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				else:
					sys.stderr.write("The RNN model is invaliad. (rnn | gru | lstm)\n")
				cur_vec = MaxPooling1D(self.time_length)(cur_vec)
				cur_vec = Flatten()(cur_vec)
			sent_model = Model(input=raw_current, output=cur_vec)

			# apply the same function for mapping word sequences into sentence vecs
			# input_memory: (, his_length, time_length, embedding_size)
			if self.input_type == 'embedding':
				raw_input_memory = Input(shape=(self.his_length * self.time_length,), dtype='int32', name='input_memory')
				input_memory = Reshape((self.his_length, self.time_length))(raw_input_memory)
			else:
				raw_input_memory = Input(shape=(self.his_length * self.time_length, self.embedding_size), name='input_memory')
				input_memory = Reshape((self.his_length, self.time_length, self.embedding_size))(raw_input_memory)
			mem_vec = TimeDistributed(sent_model)(input_memory)

			# compute the similarity between sentence embeddings for attention
			match = merge([mem_vec, cur_vec], mode='dot', dot_axes=[2, 1])
			match = Activation('softmax', name='match')(match)


			# encode the history with the current utterance and then feed into each timestep for tagging
			his_vec = merge([mem_vec, match], mode='dot', dot_axes=[1, 1])
			encoder = merge([his_vec, cur_vec], mode='sum')
			encoder = Dense(self.embedding_size)(encoder)

			# tagging the words in the current sentence
			if 'blstm' in self.arch:
				forward = LSTM(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
				backward = LSTM(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation, go_backwards=True)(current)
				labeling = merge([forward, backward], mode='concat', concat_axis=-1)
			elif 'rnn' in self.arch:
				labeling = SimpleRNN(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
			elif 'gru' in self.arch:
				labeling = GRU(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
			elif 'lstm' in self.arch:
				labeling = LSTM(self.hidden_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
			tagger = merge([encoder, labeling], mode='concat', concat_axis=-1)
			if self.dropout:
				tagger = Dropout(self.dropout_ratio)(tagger)
			prediction = Dense(self.output_vocab_size, activation='softmax')(tagger)

			self.model = Model(input=[raw_input_memory, raw_current], output=prediction)
			self.model.compile(loss='categorical_crossentropy', optimizer=opt_func)

		elif 'memn2n' in self.arch:
			# current: (, time_length, embedding_size)
			if self.input_type == 'embedding':
				raw_current = Input(shape=(self.time_length,), dtype='int32', name='raw_current')
				current = Embedding(input_dim=self.input_vocab_size, output_dim=self.embedding_size, input_length=self.time_length)(raw_current)
			else:
				current = raw_current = Input(shape=(self.time_length, self.input_vocab_size), name='current')
			if 'memn2n-d-' in self.arch:
				cur_vec = TimeDistributed(Dense(self.sembedding_size, input_shape=(self.time_length, self.embedding_size)))(current)
				cur_vec = AveragePooling1D(self.time_length)(cur_vec)
				cur_vec = Flatten()(cur_vec)
			elif 'memn2n-c-' in self.arch:
				cur_vec = Convolution1D(self.sembedding_size, 3, border_mode='same', input_shape=(self.time_length, self.embedding_size))(current)
				cur_vec = MaxPooling1D(self.time_length)(cur_vec)
				cur_vec = Flatten()(cur_vec)
			elif 'memn2n-r-' in self.arch:
				if 'blstm' in self.arch:
					fcur_vec = LSTM(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
					bcur_vec = LSTM(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation, go_backwards=True)(current)
					cur_vec = merge([fcur_vec, bcur_vec], mode='concat')
				elif 'rnn' in self.arch:
					cur_vec = SimpleRNN(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
				elif 'gru' in self.arch:
					cur_vec = GRU(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
				elif 'lstm' in self.arch:
					cur_vec = LSTM(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(current)
				else:
					sys.stderr.write("The RNN model is invaliad. (rnn | gru | lstm)\n")
			elif 'memn2n-rc-' in self.arch:
				if 'rnn' in self.arch:
					cur_vec = SimpleRNN(self.sembedding_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				elif 'gru' in self.arch:
					cur_vec = GRU(self.sembedding_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				elif 'lstm' in self.arch:
					cur_vec = LSTM(self.sembedding_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				else:
					sys.stderr.write("The RNN model is invaliad. (rnn | gru | lstm)\n")
				cur_vec = Convolution1D(self.sembedding_size, 3, border_mode='same', input_shape=(self.time_length, self.sembedding_size))(cur_vec)
				cur_vec = MaxPooling1D(self.time_length)(cur_vec)
				cur_vec = Flatten()(cur_vec)
			elif 'memn2n-cr-' in self.arch:
				cur_vec = Convolution1D(self.sembedding_size, 3, border_mode='same', input_shape=(self.time_length, self.embedding_size))(current)
				if 'rnn' in self.arch:
					cur_vec = SimpleRNN(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				elif 'gru' in self.arch:
					cur_vec = GRU(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				elif 'lstm' in self.arch:
					cur_vec = LSTM(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				else:
					sys.stderr.write("The RNN model is invaliad. (rnn | gru | lstm)\n")
			elif 'memn2n-crp-' in self.arch:
				cur_vec = Convolution1D(self.sembedding_size, 3, border_mode='same', input_shape=(self.time_length, self.embedding_size))(current)
				if 'rnn' in self.arch:
					cur_vec = SimpleRNN(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				elif 'gru' in self.arch:
					cur_vec = GRU(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				elif 'lstm' in self.arch:
					cur_vec = LSTM(self.sembedding_size, return_sequences=False, init=self.init_type, activation=self.activation)(cur_vec)
				else:
					sys.stderr.write("The RNN model is invaliad. (rnn | gru | lstm)\n")
				cur_vec = MaxPooling1D(self.time_length)(cur_vec)
				cur_vec = Flatten()(cur_vec)
			sent_model = Model(input=raw_current, output=cur_vec)

			# apply the same function for mapping word sequences into sentence vecs
			# input_memory: (, his_length, time_length, embedding_size)
			if self.input_type == 'embedding':
				raw_input_memory = Input(shape=(self.his_length * self.time_length,), dtype='int32', name='input_memory')
				input_memory = Reshape((self.his_length, self.time_length))(raw_input_memory)
			else:
				raw_input_memory = Input(shape=(self.his_length * self.time_length, self.embedding_size), name='input_memory')
				input_memory = Reshape((self.his_length, self.time_length, self.embedding_size))(raw_input_memory)
			mem_vec = TimeDistributed(sent_model)(input_memory)

			# compute the similarity between sentence embeddings for attention
			match = merge([mem_vec, cur_vec], mode='dot', dot_axes=[2, 1])
			match = Activation('softmax', name='match')(match)


			# encode the history with the current utterance and then feed into each timestep for tagging
			his_vec = merge([mem_vec, match], mode='dot', dot_axes=[1, 1])
			encoder = merge([his_vec, cur_vec], mode='sum')
			encoder = Dense(self.embedding_size)(encoder)
			encoder = RepeatVector(self.time_length)(encoder)

			# tagging the words in the current sentence
			if 'blstm' in self.arch:
				forward = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
				backward = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation, go_backwards=True)(current)
				labeling = merge([forward, backward], mode='concat', concat_axis=-1)
			elif 'rnn' in self.arch:
				labeling = SimpleRNN(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
			elif 'gru' in self.arch:
				labeling = GRU(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
			elif 'lstm' in self.arch:
				labeling = LSTM(self.hidden_size, return_sequences=True, init=self.init_type, activation=self.activation)(current)
			tagger = merge([encoder, labeling], mode='concat', concat_axis=-1)
			if self.dropout:
				tagger = Dropout(self.dropout_ratio)(tagger)
			prediction = TimeDistributed(Dense(self.output_vocab_size, activation='softmax'))(tagger)

			self.model = Model(input=[raw_input_memory, raw_current], output=prediction)
			self.model.compile(loss='categorical_crossentropy', optimizer=opt_func)


	def train(self, H_train, X_train, y_train, H_dev, X_dev, y_dev, val_ratio=0.0):
		# load saved model weights
		if self.load_weight is not None:
			sys.stderr.write("Load the pretrained weights for the model.\n")
			self.model.load_weights(self.load_weight)
		else:
			# training batch preparation
			if 'memn2n' in self.arch or self.arch[0] == 'h':
				batch_train = [H_train, X_train]
				batch_dev = [H_dev, X_dev]
			else:
				batch_train = X_train
				batch_dev = X_dev
			# model training
			if not self.nodev:
				early_stop = EarlyStopping(monitor='val_loss', patience=10)
				train_log = LossHistory()
				self.model.fit(batch_train, y_train, batch_size=self.batch_size, nb_epoch=self.max_epochs, verbose=1, validation_data=(batch_dev, y_dev), callbacks=[early_stop, train_log], shuffle=self.shuffle)
				if self.log is not None:
					fo = open(self.log, "wb")
					for loss in train_log.losses:
						fo.write("%lf\n" %loss)
					fo.close()
			else:
				if self.set_batch:
					cur = 0
					total = np.sum(np.array(self.trainNum))
					for num in self.trainNum:
						print("Current batch is: %d ( %.1f%% )\r" %(cur, float(cur) / float(total) * 100)),
						sys.stdout.flush()
						single_batch_train = [single_batch[cur:cur+num+1][:] for single_batch in batch_train]
						for iter_epoch in range(0, self.max_epochs):
							self.model.train_on_batch(single_batch_train, y_train[cur:cur+num+1])
						cur += num + 1
				else:
					self.model.fit(batch_train, y_train, batch_size=self.batch_size, nb_epoch=self.max_epochs, verbose=1, shuffle=self.shuffle)

	def run(self):
		# initialization of vocab
		emptyVocab = {}
		emptyIndex = list()
		trainData = dataSet(self.training_file,'train',emptyVocab,emptyVocab,emptyIndex,emptyIndex)
		testData = dataSet(self.test_file, 'test', trainData.getWordVocab(), trainData.getTagVocab(),trainData.getIndex2Word(),trainData.getIndex2Tag())
		if self.train_numfile is not None:
			self.trainNum, self.trainTotal = readNum(self.train_numfile)
			self.testNum, self.testTotal = readNum(self.test_numfile)
			if not self.nodev:
				self.devNum, self.devTotal = readNum(self.dev_numfile)

		target_file = self.result_path + '/' + 'tag.list'
		fo = open(target_file, "wb")
		for tag in trainData.dataSet['id2tag']:
			fo.write("%s\n" %tag)
		fo.close()

		# preprocessing by padding 0 until maxlen
		pad_X_train = sequence.pad_sequences(trainData.dataSet['utterances'], maxlen=self.time_length, dtype='int32', padding='pre')
		pad_X_test = sequence.pad_sequences(testData.dataSet['utterances'], maxlen=self.time_length, dtype='int32', padding='pre')
		pad_y_train = sequence.pad_sequences(trainData.dataSet['tags'], maxlen=self.time_length, dtype='int32', padding='pre')
		num_sample_train, max_len = np.shape(pad_X_train)
		num_sample_test, max_len = np.shape(pad_X_test)
		
		if not self.nodev:
			validData = dataSet(self.validation_file, 'val', trainData.getWordVocab(), trainData.getTagVocab(),trainData.getIndex2Word(),trainData.getIndex2Tag())
			pad_X_dev = sequence.pad_sequences(validData.dataSet['utterances'], maxlen=self.time_length, dtype='int32', padding='pre')
			pad_y_dev = sequence.pad_sequences(validData.dataSet['tags'], maxlen=self.time_length, dtype='int32', padding='pre')
			num_sample_dev, max_len = np.shape(pad_X_dev)

		# encoding input vectors
		self.input_vocab_size = trainData.getWordVocabSize()
		self.output_vocab_size = trainData.getTagVocabSize()

		# building model architecture
		self.build()

		# data generation
		sys.stderr.write("Vectorizing the input.\n")
		X_train = encoding(pad_X_train, self.input_type, self.time_length, self.input_vocab_size)
		X_test = encoding(pad_X_test, self.input_type, self.time_length, self.input_vocab_size)
		y_train = encoding(pad_y_train, '1hot', self.time_length, self.output_vocab_size)

		if not self.nodev:
			X_dev = encoding(pad_X_dev, self.input_type, self.time_length, self.input_vocab_size)
			y_dev = encoding(pad_y_dev, '1hot', self.time_length, self.output_vocab_size)

		if 'memn2n' in self.arch or self.arch[0] == 'h':
			# encode history for memory network
			pad_H_train = sequence.pad_sequences(history_build(trainData, pad_X_train), maxlen=(self.time_length * self.his_length), dtype='int32', padding='pre')
			pad_H_test = sequence.pad_sequences(history_build(testData, pad_X_test), maxlen=(self.time_length * self.his_length), dtype='int32', padding='pre')
			H_train = encoding(pad_H_train, self.input_type, self.time_length * self.his_length, self.input_vocab_size)
			H_test = encoding(pad_H_test, self.input_type, self.time_length * self.his_length, self.input_vocab_size)
			if not self.nodev:
				pad_H_dev = sequence.pad_sequences(history_build(validData, pad_X_dev), maxlen=(self.time_length * self.his_length), dtype='int32', padding='pre')
				H_dev = encoding(pad_H_dev, self.input_type, self.time_length * self.his_length, self.input_vocab_size)
			if self.e2e_flag:
				H_train = np.array([H_train[num - 1] for num in self.trainTotal])
				X_train = np.array([X_train[num - 1] for num in self.trainTotal])
				y_train = np.array([y_train[num - 1] for num in self.trainTotal])
				pad_X_test = np.array([pad_X_test[num - 1] for num in self.testTotal])
				H_test = np.array([H_test[num - 1] for num in self.testTotal])
				X_test = np.array([X_test[num - 1] for num in self.testTotal])
				if not self.nodev:
					pad_X_dev = np.array([pad_X_dev[num - 1] for num in self.devTotal])
					H_dev = np.array([H_dev[num - 1] for num in self.devTotal])
					X_dev = np.array([X_dev[num - 1] for num in self.devTotal])
					y_dev = np.array([y_dev[num - 1] for num in self.devTotal])
		else:
			H_train = H_test = H_dev = None

		if self.nodev:
			H_dev = X_dev = y_dev = None

		if self.record_epoch != -1 and self.load_weight is None:
			total_epochs = self.max_epochs
			self.max_epochs = self.record_epoch
			for i in range(1, total_epochs / self.record_epoch + 1):
				num_iter = i * self.record_epoch
				self.train(H_train=H_train, X_train=X_train, y_train=y_train, H_dev=H_dev, X_dev=X_dev, y_dev=y_dev)
				if not self.nodev:
					self.test(H=H_dev, X=X_dev, data_type='dev.'+str(num_iter),tagDict=trainData.dataSet['id2tag'], pad_data=pad_X_dev)
				self.test(H=H_test, X=X_test, data_type='test.'+str(num_iter), tagDict=trainData.dataSet['id2tag'], pad_data=pad_X_test)
				# save weights for the current model
				whole_path = self.mdl_path + '/' + self.model_arch + '.' + str(num_iter) + '.h5'
				sys.stderr.write("Writing model weight to %s...\n" %whole_path)
				self.model.save_weights(whole_path, overwrite=True)
		else:
			self.train(H_train=H_train, X_train=X_train, y_train=y_train, H_dev=H_dev, X_dev=X_dev, y_dev=y_dev)
			if not self.nodev:
				self.test(H=H_dev, X=X_dev, data_type='dev', tagDict=trainData.dataSet['id2tag'], pad_data=pad_X_dev)
			self.test(H=H_test, X=X_test, data_type='test', tagDict=trainData.dataSet['id2tag'], pad_data=pad_X_test)
			if self.load_weight is None:
				whole_path = self.mdl_path + '/' + self.model_arch + '.final-' + str(self.max_epochs) + '.h5'
				sys.stderr.write("Writing model weight to %s...\n" %whole_path)
				self.model.save_weights(whole_path, overwrite=True)
