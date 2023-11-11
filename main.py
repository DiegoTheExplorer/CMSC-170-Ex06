#Name: Diego Miguel M. Villamil
#Section: CMSC 170 WX-3L

fp = open("input.txt", "r")

lr = float(fp.readline()) #Learning Rate
th = float(fp.readline()) #Threshold
bias = float(fp.readline()) 
numX = 0

print("Learning Rate: ",lr)
print("Threshold: ",th)
print("Bias: ", bias)

pTable = []
iterations = []

for line in fp: #Read the feature vectors and expected output from input.txt
  line = line.split(" ")
  line = list(map(float,line))
  numX = len(line) - 1
  line.insert(len(line) - 1, bias)

  for ind in range(0,(numX + 2)): #Add 0s for weights, perceptron value (a) and classification (y)
    line.insert(len(line) - 1, float(0))
  pTable.append(line)
fp.close()
pTable.append([])
for ind in range(0,len(pTable[0])):
  pTable[len(pTable) - 1].append(0)

ln = len(pTable[0])
a = ln - 3
y = ln - 2
z = ln - 1
for row in range(0,len(pTable) - 1):
  #Compute perceptron value a
  for i in range(0,(numX)):
    pTable[row][a] = pTable[row][a] + (pTable[row][i] * pTable[row][i + numX])

  #Determine classification
  pTable[row][y] = 1 if pTable[row][a] > th else 0

  #Adjust weights
  for i in range(0,(numX + 1)):
    pTable[row + 1][i + numX] = pTable[row][i + numX] + (lr * pTable[row][i] * (pTable[row][z] - pTable[row][y]))
  print(pTable[row])

