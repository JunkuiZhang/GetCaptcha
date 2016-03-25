from PIL import Image
import os
import csv
import numpy


class GetCaptchaTrainingData:

	def __init__(self):
		self.captcha_image_list = self.get_captcha_list()
		self.captcha_string_list = self.get_captcha_strings()
		self.get_white_black_image()
		self.get_blocks_from_image()

	def get_captcha_list(self):
		fs = os.listdir("./Data/")
		image_list = []
		for f in fs:
			if f.endswith(".jpg"):
				image_list.append(f)
		print(image_list)
		return image_list

	def get_captcha_strings(self):
		f = open("./Data/captcha_info.csv")
		w = csv.reader(f)
		captcha_string_list = []
		for line in w:
			captcha_string_list.append(line[0])
		return captcha_string_list

	def get_white_black_image(self):
		for f in self.captcha_image_list:
			img = Image.open("./Data/" + str(f))
			img.convert("RGB")
			pixes_data = img.load()
			for x in range(img.size[0]):
				for y in range(img.size[1]):
					if pixes_data[x, y][0] >= 160 and sum(pixes_data[x, y]) < 350:
						pixes_data[x, y] = (0, 0, 0)
					else:
						pixes_data[x, y] = (255, 255, 255)
			img.save("./Data/train/%s" % f)

	def get_blocks_from_image(self):
		num = 0
		for f in os.listdir("./Data/train/"):
			image = Image.open("./Data/train/" + f)
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
				single_string.save("./Data/train_data/%s" % (str(num)+".jpg"))


class NeuralNetwork:

	def __init__(self):
		self.num_examples = len(os.listdir("./Data/train_data/"))
		self.nn_input_dim = 2
		self.nn_output_dim = 2
		self.epsilon = 0.01
		self.reg_lambda = 0.01
		self.nn_hdim = 5
		self.X = self.convert_single_string_to_array("./Data/train_data/1.jpg")
		# self.convert_single_string_to_array("./Data/train_data/1.jpg")

	def convert_single_string_to_array(self, path):
		image = Image.open(path)
		pixes = image.load()
		array = []
		for i in range(7):
			for j in range(10):
				num = 0
				if pixes[2*i, 2*j] == (0, 0, 0):
					num += 1
				if pixes[2*i + 1, 2*j] == (0, 0, 0):
					num += 1
				if pixes[2*i, 2*j + 1] == (0, 0, 0):
					num += 1
				if pixes[2*i + 1, 2*j + 1] == (0, 0, 0):
					num += 1
				array.append(num)
		return numpy.array(array)

	def calculate_loss(self, model, X):
		W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
		z1 = X.dot(W1) + b1
		a1 = numpy.tanh(z1)
		z2 = a1.dot(W2) + b2
		exp_scores = numpy.exp(z2)
		probs = exp_scores / numpy.sum(exp_scores, axis=1, keepdims=True)
		correct_logprobs = -numpy.log(probs[range(self.num_examples)])

	def build_model(self, num_passes = 20000, print_loss = False):
		W1 = numpy.random.randn(self.nn_input_dim, self.nn_hdim) / numpy.sqrt(self.nn_input_dim)
		b1 = numpy.zeros((1, self.nn_hdim))
		W2 = numpy.random.randn(self.nn_hdim, self.nn_output_dim) / numpy.sqrt(self.nn_hdim)
		b2 = numpy.zeros((1, self.nn_hdim))

		for i in range(0, num_passes):
			z1 = self.X.dot(W1) + b1
			a1 = numpy.tanh(z1)
			z2 = a1.dot(W2) + b2
			exp_scores = numpy.exp(z2)
			probs = exp_scores / numpy.sum(exp_scores, axis=1, keepdims=True)

			delta3 = probs
			delta3[range(self.num_examples)] -= 1
			dW2 = (a1.T).dot(delta3)
			db2 = numpy.sum(delta3, axis=0, keepdims=True)
			delta2 = delta3.dot(W2.T) * (1 - numpy.power(a1, 2))
			dW1 = numpy.dot(self.X.T, delta2)
			db1 = numpy.sum(delta2, axis=0)

			dW2 += self.reg_lambda * W2
			dW1 += self.reg_lambda * W1

			W1 += -self.epsilon * dW1
			b1 += -self.epsilon * db1
			W2 += -self.epsilon * dW2
			b2 += -self.epsilon * db2

			model = {
				"W1": W1,
				"b1": b1,
				"W2": W2,
				"b2": b2
			}

			if print_loss and i % 1000 == 0:
				print("Loss after iteration %i: %f" %(i, self.calculate_loss(model)))


s = GetCaptchaTrainingData()
n = NeuralNetwork()