from PIL import Image
import os
import csv
import numpy
import requests
import pickle


class GetCaptchaData:

	def __init__(self, training):
		self.state = training
		if self.state:
			self.captcha_image_list = self.get_captcha_list()
			self.captcha_string_list = self.get_captcha_strings()
			self.get_white_black_image(self.captcha_image_list, "./Data/")
			self.get_blocks_from_image("")

	def get_captcha_list(self):
		fs = os.listdir("./Data/")
		image_list = []
		for f in fs:
			if f.endswith(".jpg"):
				image_list.append(f)
		return image_list

	def get_captcha_strings(self):
		f = open("./Data/captcha_info.csv")
		w = csv.reader(f)
		captcha_string_list = []
		for line in w:
			captcha_string_list.append(line[0])
		return captcha_string_list

	def get_white_black_image(self, file_list, path):
		if self.state:
			for f in file_list:
				img = Image.open(path + str(f))
				img.convert("RGB")
				pixes_data = img.load()
				for x in range(img.size[0]):
					for y in range(img.size[1]):
						if pixes_data[x, y][0] >= 160 and sum(pixes_data[x, y]) < 350:
							pixes_data[x, y] = (0, 0, 0)
						else:
							pixes_data[x, y] = (255, 255, 255)
				img.save(path + "/train/%s" % f)
		else:
			img = Image.open(path)
			img.convert("RGB")
			pixes_data = img.load()
			for x in range(img.size[0]):
				for y in range(img.size[1]):
					if pixes_data[x, y][0] >= 160 and sum(pixes_data[x, y]) < 350:
						pixes_data[x, y] = (0, 0, 0)
					else:
						pixes_data[x, y] = (255, 255, 255)
			return img

	def get_blocks_from_image(self, img):

		def working(image, num):
			pixes_data = image.load()
			min_x = 0
			max_x = 0
			min_y = 0
			for i in range(image.size[0]):
				for j in range(image.size[1]):
					if pixes_data[i, j] == (0, 0, 0):
						min_x = i
						break
				if min_x != 0:
					break
			for i in reversed(range(image.size[0])):
				for j in range(image.size[1]):
					if pixes_data[i, j] == (0, 0, 0):
						max_x = i
						break
				if max_x != 0:
					break
			for j in reversed(range(image.size[1])):
				for i in range(image.size[0]):
					if pixes_data[i, j] == (0, 0, 0):
						min_y = j
						break
				if min_y != 0:
					break
			single_s = []
			for n in range(5):
				single_s.append(image.crop((min_x + 14 * n, min_y - 20, min_x + 14 * (n + 1), min_y)))
			for single_string in single_s:
				num += 1
				if self.state:
					single_string.save("./Data/train_data/%s" % (str(num)+".jpg"))
				else:
					single_string.save("./Data/test_data/%s" % (str(num) + ".jpg"))

		if self.state:
			num = 0
			for f in os.listdir("./Data/train/"):
				image = Image.open("./Data/train/" + f)
				working(image, num)
		else:
			working(img, 0)




class NeuralNetwork:

	def __init__(self):
		self.num_examples = len(os.listdir("./Data/train_data/"))
		self.epsilon = 0.01
		self.reg_lambda = 0.01
		self.nn_hdim = 80
		self.X = self.get_x(path="./Data/train_data/")
		self.captcha_stuff = GetCaptchaData(training=True)
		self.y = self.get_y()
		self.nn_input_dim = len(self.X[0])
		self.nn_output_dim = len(self.y[0])

	def get_x(self, path):
		x = []
		for f in os.listdir(path):
			x.append(self.convert_single_string_to_array(path + f))
		return numpy.array(x)

	def convert_single_string_to_array(self, path):
		image = Image.open(path)
		pixes = image.load()
		array = []
		for i in range(14):
			for j in range(20):
				num = 0
				if pixes[i, j] == (0, 0, 0):
					num += 1
				elif pixes[i, j] == (255, 255, 255):
					num += 1
				array.append(num)
		return array

	def calculate_loss(self, o2, y):
		loss = 0.5 * numpy.square(o2 - y)
		return numpy.sum(loss)

	def get_captcha_strings_y(self):
		captcha_list = self.captcha_stuff.captcha_string_list
		l = []
		for captcha in captcha_list:
			for s in captcha:
				if s not in l:
					l.append(s)
		return l

	def get_y(self):
		captcha_list = self.get_captcha_strings_y()
		captcha = self.captcha_stuff.captcha_string_list
		y = []
		for cap in captcha:
			for s in cap:
				l = []
				for n in range(len(captcha_list)):
					l.append(0)
				l[captcha_list.index(s)] = 1
				y.append(l)
		return numpy.array(y)

	def build_model(self, num_passes=20000, print_loss=False):

		def sigmoid(x, deriv=False):
			if deriv:
				return x * (1 - x)
			return 1 / (1 + numpy.exp(-x))

		W1 = 0.2 * numpy.random.random((self.nn_input_dim, self.nn_hdim)) - 0.1
		b1 = numpy.zeros((1, self.nn_hdim))
		W2 = 0.2 * numpy.random.random((self.nn_hdim, self.nn_output_dim)) - 0.1
		b2 = numpy.zeros((1, self.nn_output_dim))

		for i in range(0, num_passes):
			# print(i)
			# print("=="*20)
			# for n in range(len(self.X)):
			for n in range(len(self.X)):
				x = self.X[n]
				y = self.y[n]

				i1 = numpy.dot(x, W1)
				o1 = sigmoid(i1 + b1)
				i2 = numpy.dot(o1, W2)
				o2 = sigmoid(i2 + b2)

				l2_delta = sigmoid(o2, deriv=True) * (o2 - y)
				l1_delta = l2_delta.dot(W2.T) * sigmoid(o1, deriv=True)

				# W1 -= numpy.dot(x.T, l1_delta)
				W1 -= 0.2 * x.T.reshape(self.nn_input_dim, 1) @ l1_delta.reshape(1, self.nn_hdim)
				W2 -= 0.2 * (o1.T).dot(l2_delta)
				b1 -= 0.2 * l1_delta
				b2 -= 0.2 * l2_delta
			# x = self.X
			# y = self.y
			# l1 = sigmoid(numpy.dot(x, W1))
			# l2 = sigmoid(numpy.dot(l1, W2))
			#
			# l2_error = l2 - y
			# l2_delta = l2_error * sigmoid(l2, deriv=True)
			# l1_delta = l2_delta.dot(W2.T) * sigmoid(l1, deriv=True)
			# W2 -= 0.5 * (l1.T).dot(l2_delta)
			# W1 -= 0.5 * (x.T).dot(l1_delta)

			model = {
				"W1": W1,
				"b1": b1,
				"W2": W2,
				"b2": b2
			}

			if print_loss and i % 1000 == 0:
				print("Loss after iteration %i: %s" %(i, str(self.calculate_loss(o2, y))))
		f = open("net.txt", "wb")
		pickle.dump(model, f)
		f.close()
		return model

	def run(self):

		def sigmoid(x):
			return 1 / (1 + numpy.exp(-x))

		if not os.path.exists("./net.txt"):
			model = self.build_model(print_loss=True)
		else:
			f = open("net.txt", "rb")
			model = pickle.load(f)
			f.close()
		# model = self.build_model()
		session = requests.session()
		res = session.get("http://210.42.121.241/servlet/GenImg")
		print("Done")
		f = open("test.jpg", "wb")
		f.write(res.content)
		f.close()

		data = GetCaptchaData(training=False)
		img = data.get_white_black_image("", "test.jpg")
		data.get_blocks_from_image(img)

		X = self.get_x("./Data/test_data/")
		out = []
		for x in X:
			out.append(sigmoid(x.dot(model["W1"]) + model["b1"]).dot(model["W2"]) + model["b2"])
		c = self.get_captcha_strings_y()
		ca = []
		print(len(out))
		for o in out:
			m = numpy.argmax(o[0])
			ca.append(c[m])
		print(ca)


n = NeuralNetwork()
n.run()
# n.build_model()