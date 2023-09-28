import re, sys, os
from collections import Counter
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import re, sys, os, platform
import math
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import numpy as np
import pandas as pd


def readFasta(file):
	if os.path.exists(file) == False:
		print('Error: "' + file + '" does not exist.')
		sys.exit(1)

	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input file seems not in fasta format.')
		sys.exit(1)

	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta

def checkFasta(fastas):
	status = True
	lenList = set()
	for i in fastas:
		lenList.add(len(i[1]))
	if len(lenList) == 1:
		return True
	else:
		return False

def minSequenceLength(fastas):
	minLen = 10000
	for i in fastas:
		if minLen > len(i[1]):
			minLen = len(i[1])
	return minLen

def minSequenceLengthWithNormalAA(fastas):
	minLen = 10000
	for i in fastas:
		if minLen > len(re.sub('-', '', i[1])):
			minLen = len(re.sub('-', '', i[1]))
	return minLen

def AAC(fastas, **kw):
	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	#AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#']
	for i in AA:
		header.append(i)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		count = Counter(sequence)
		for key in count:
			count[key] = count[key]/len(sequence)
		code = [name]
		for aa in AA:
			code.append(count[aa])
		encodings.append(code)
	return encodings

def CKSAAP(fastas, gap=5, **kw):
	if gap < 0:
		print('Error: the gap should be equal or greater than zero' + '\n\n')
		return 0

	if minSequenceLength(fastas) < gap+2:
		print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap+2) + '\n\n')
		return 0

	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	encodings = []
	aaPairs = []
	for aa1 in AA:
		for aa2 in AA:
			aaPairs.append(aa1 + aa2)
	header = ['#']
	for g in range(gap+1):
		for aa in aaPairs:
			header.append(aa + '.gap' + str(g))
	encodings.append(header)
	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		for g in range(gap+1):
			myDict = {}
			for pair in aaPairs:
				myDict[pair] = 0
			sum = 0
			for index1 in range(len(sequence)):
				index2 = index1 + g + 1
				if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
					myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
					sum = sum + 1
			for pair in aaPairs:
				code.append(myDict[pair] / sum)
		encodings.append(code)
	return encodings


def Rvalue(aa1, aa2, AADict, Matrix):
	return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def PAAC(fastas, lambdaValue=6, w=0.5, **kw):
	if minSequenceLengthWithNormalAA(fastas) < lambdaValue + 1:
		print('Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
		return 0

	dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/data/PAAC.txt'
	with open(dataFile) as f:
		records = f.readlines()
	AA = ''.join(records[0].rstrip().split()[1:])
	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i
	AAProperty = []
	AAPropertyNames = []
	for i in range(1, len(records)):
		array = records[i].rstrip().split() if records[i].rstrip() != '' else None
		AAProperty.append([float(j) for j in array[1:]])
		AAPropertyNames.append(array[0])

	AAProperty1 = []
	for i in AAProperty:
		meanI = sum(i) / 20
		fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
		AAProperty1.append([(j-meanI)/fenmu for j in i])

	encodings = []
	header = ['#']
	for aa in AA:
		header.append('Xc1.' + aa)
	for n in range(1, lambdaValue + 1):
		header.append('Xc2.lambda' + str(n))
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		theta = []
		for n in range(1, lambdaValue + 1):
			theta.append(
				sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
				len(sequence) - n))
		myDict = {}
		for aa in AA:
			myDict[aa] = sequence.count(aa)
		code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
		code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
		encodings.append(code)
	return encodings

kw = {'path': r"C:",'order': 'ACDEFGHIKLMNPQRSTVWY'}
fastas = readFasta(r"./data/test dataset/secondStage/secondStage_test.faa")
aac = AAC(fastas, **kw)
cksaap = CKSAAP(fastas, 5, **kw)
paac = PAAC(fastas, 4, 0.05,  **kw)

data_AAC = np.matrix(aac[1:])[:, 1:]
data_AAC = pd.DataFrame(data=data_AAC)

data_CKSAAP = np.matrix(cksaap[1:])[:, 1:]
data_CKSAAP = pd.DataFrame(data=data_CKSAAP)

data_PAAC = np.matrix(paac[1:])[:, 1:]
data_PAAC = pd.DataFrame(data=data_PAAC)

feature = np.column_stack((data_AAC, data_CKSAAP, data_PAAC))
feature = pd.DataFrame(feature)
print(feature)
feature.to_csv(r"./data/test dataset/secondStage/feature.csv", header=None, index=False)
