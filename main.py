#Name: Diego Miguel M. Villamil
#Section: CMSC 170 WX-3L

from decimal import *
from copy import deepcopy

fp = open("input.txt", "r")

lr = Decimal(fp.readline()) #Learning Rate
th = Decimal(fp.readline()) #Threshold
bias = Decimal(fp.readline()) 
numX = 0
Converging = False

print("Learning Rate: ",lr)
print("Threshold: ",th)
print("Bias: ", bias)

pTable = []
iterations = []

for line in fp: #Read the feature vectors and expected output from input.txt
  line = line.split(" ")
  line = list(map(Decimal,line))
  numX = len(line) - 1
  line.insert(len(line) - 1, bias)

  for ind in range(0,(numX + 3)): #Add 0s for weights, perceptron value (a) and classification (y)
    line.insert(len(line) - 1, Decimal(0))
  pTable.append(line)
fp.close()
pTable.append([])
for ind in range(0,len(pTable[0])):
  pTable[len(pTable) - 1].append(Decimal(0))
ln = len(pTable[0])
a = ln - 3
y = ln - 2
z = ln - 1

while(not Converging and (len(iterations) < 1000)):
  Converging = True
  for row in range(0,len(pTable) - 1):
    #Compute perceptron value a
    for i in range(0,(numX + 1)):
      feat = pTable[row][i]
      weight = pTable[row][i + numX + 1]
      aVal = feat * weight
      currA = pTable[row][a]
      pTable[row][a] = currA + aVal

    #Determine classification
    pTable[row][y] = 1 if pTable[row][a] > th else 0

    #Adjust weights
    for i in range(0,(numX + 1)):
      currW = pTable[row][i + numX + 1]
      inpVal = pTable[row][i]
      expected = pTable[row][z]
      predicted = pTable[row][y]
      newW = currW + (lr * inpVal * (expected - predicted))
      pTable[row + 1][i + numX + 1] = newW

  iterations.append(deepcopy(pTable))

  #Check for convergence
  for row in range(1,len(pTable) - 1):
    for i in range(0,(numX + 1)):
      if(pTable[row][i + numX + 1] != pTable[row + 1][i + numX + 1]):
        Converging = False
        break
    if(not Converging):
      break

  #Weights for the last row of current iteration become the starting weights for the next iteration    
  tempTable = deepcopy(pTable)
  for row in range(0,len(tempTable)):
    for col in range((numX + 1),len(tempTable[row]) - 1):
      tempTable[row][col] = Decimal(0)

  for i in range(0,(numX + 1)): 
    tempTable[0][i + numX + 1] = pTable[len(pTable) - 1][i + numX + 1]
  
  pTable = deepcopy(tempTable)

#Creating the label array
labelArr = []
for i in range(0,numX):
  temp = "x" + str(i)
  labelArr.append(temp)
labelArr.append("b")
for i in range(0,numX):
  temp = "w" + str(i)
  labelArr.append(temp)
labelArr.append("wb")
labelArr.append("a")
labelArr.append("y")
labelArr.append("z")

for itr in iterations:
  itr.insert(0,labelArr)

#Write iterations to output.txt
fp = open("output.txt", "w")
for itr in iterations:
  temp = "Iteration " + str(iterations.index(itr) + 1) + ":\n"
  fp.write(temp)
  for row in itr:
    for val in row:
      temp = 13 * " " + str(val) + " " * (4 - (len(str(val))))
      fp.write(temp)
    fp.write("\n")
  fp.write("\n")
fp.close()