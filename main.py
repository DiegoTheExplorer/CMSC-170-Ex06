#Name: Diego Miguel M. Villamil
#Section: CMSC 170 WX-3L

fp = open("input.txt", "r")

lr = float(fp.readline()) #Learning Rate
th = float(fp.readline()) #Threshold
bias = float(fp.readline()) 
numX = 0
Converging = False

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

  for ind in range(0,(numX + 3)): #Add 0s for weights, perceptron value (a) and classification (y)
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

while(not Converging and (len(iterations) < 1000)):
  print("Iteration: ", len(iterations) + 1)
  Converging = True
  for row in range(0,len(pTable) - 1):
    #Compute perceptron value a
    for i in range(0,(numX + 1)):
      feat = pTable[row][i]
      weight = pTable[row][i + numX + 1]
      aVal = feat * weight
      pTable[row][a] = round(pTable[row][a] + aVal,2)

    #Determine classification
    pTable[row][y] = 1 if pTable[row][a] > th else 0

    #Adjust weights
    for i in range(0,(numX + 1)):
      currW = pTable[row][i + numX + 1]
      inpVal = pTable[row][i]
      expected = pTable[row][z]
      predicted = pTable[row][y]
      newW = currW + (lr * inpVal * (expected - predicted))
      pTable[row + 1][i + numX + 1] = round(newW,2)
    print(pTable[row])

  print(pTable[len(pTable) - 1])
  print()
  iterations.append(pTable.copy())

  #Check for convergence
  for row in range(1,len(pTable) - 1):
    for i in range(0,(numX + 1)):
      if(pTable[row][i + numX + 1] != pTable[row + 1][i + numX + 1]):
        Converging = False
        break
    if(not Converging):
      break

  for i in range(0,(numX + 1)):
    pTable[0][i + numX + 1] = pTable[len(pTable) - 1][i + numX + 1]
