import json


with open('./result.json', 'r', encoding='utf-8-sig') as f:
    res = json.loads(f.read())

totalKey = '总的来说'

totalValue = res[totalKey]


otherValue = []

for itemKey in res:
    if itemKey != totalKey:
        otherValue.append(res[itemKey])

print(totalValue)
print(otherValue)

sumAcc = 0
sumRecall = 0
sumF1 = 0
for o in otherValue:
    sumAcc += o['accuracy']
    sumRecall += o['weighted avg']['recall']
    sumF1 += o['weighted avg']['f1-score']

print('accuracy', sumAcc / len(otherValue), totalValue['accuracy'])
print('recall', sumRecall / len(otherValue), totalValue['weighted avg']['recall'])
print('f1-score', sumF1 / len(otherValue), totalValue['weighted avg']['f1-score'])
