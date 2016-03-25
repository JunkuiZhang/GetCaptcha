import csv

f = open("./Data/captcha_info.csv")
file_data = csv.reader(f)
strings = []
for line in file_data:
	strings.append(line[0])

all_s = "ABCDEFGHIJKLMNOPQLSTUVWXYZ"
all_n = "1234567890"
res = {}
for s in all_s.lower():
	res["%s" % s] = 0
for number in all_n:
	res["%s" % str(number)] = 0


def get_id(l, _id):
	for thing in l:
		if thing["id"] == _id:
			print(l.index(thing))
			print(l[l.index(thing)])
			return [l.index(thing), thing["num"]]

print(strings)
for captcha in strings:
	# for S in all_s:
	# 	if S in captcha:
	# 		_index, num = get_id(res, S)
	# 		# res[res.index(_index)]["num"] = num + 1
	for s in all_s.lower():
		if s in captcha:
			res["%s" % s] += 1
	for number in all_n:
		if number in captcha:
			res["%s" % str(number)] += 1


def key_func(asd):
	return asd[1]

print(sorted(res.items(), key=key_func, reverse=True))