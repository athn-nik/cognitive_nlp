#Extraxt semantic features fi(w)
#!/usr/bin/env python
import os
import sys
import re
from sys import argv
import pickle
import numpy as np

def find_verb(arg):
    switcher = {
        "see":0,
        "say":1,
        "taste":2,
        "wear":3,
        "open":4,
        "run":5,
        "neared":6,
        "eat":7,
        "hear":8,
        "drive":9,
        "ride":10,
        "touch":11,
        "break":12,
        "enter":13,
        "move":14,
        "listen":15,
        "approach":16,
        "fill":17,
        "clean":18,
        "lift":19,
        "rub":20,
        "smell":21,
        "fear":22,
        "push":23,
        "manipulate":24,
    }
    return switcher.get(arg, "nothing")

#script, txt = argv
verbs=[['see' ,'sees' ],['say' ,'said', 'says'],['taste' ,'tasted', 'tastes'],
['wear' ,'wore', 'wears'] ,['open' ,'opens' 'opened'] ,['run' ,'ran' 'runs'] ,
['neared', 'nears' 'near'] ,['eat' ,'ate' ,'eats'] ,['hear' ,'hears' ,'heard'],
['drive' ,'drove' ,'drives'],['ride' ,'rides' ,'rode'] ,['touch' ,'touched', 'touches'],
['break' ,'broke', 'breaks'] ,['enter' ,'entered', 'enters'],['move' ,'moved' ,'moves'],
['listen' ,'listens' ,'listened'],['approach' ,'approaches' ,'approached'],
['fill', 'filled' ,'fills'],['clean', 'cleaned' ,'cleans'] ,['lift', 'lifted' ,'lifts'] ,
['rub' ,'rubbed', 'rubs'] ,['smell', 'smells' ,'smelled'] ,['fear' ,'fears' ,'feared'] ,
['push' ,'pushed' ,'pushes'],['manipulate' ,'manipulates' ,'manipulated']]

cte_no=['airplane', 'ant', 'apartment', 'arch', 'arm', 'barn', 'bear', 'bed', 'bee',
'beetle', 'bell', 'bicycle', 'bottle', 'butterfly', 'car', 'carrot', 'cat', 'celery',
'chair', 'chimney', 'chisel', 'church', 'closet', 'coat', 'corn', 'cow', 'cup', 'desk',
'dog', 'door', 'dress', 'dresser', 'eye', 'fly', 'foot', 'glass', 'hammer', 'hand',
'horse', 'house', 'igloo', 'key', 'knife', 'leg', 'lettuce', 'pants', 'pliers',
'refrigerator', 'saw', 'screwdriver', 'shirt', 'skirt', 'spoon', 'table', 'telephone',
'tomato', 'train', 'truck', 'watch', 'window']

for x in verbs:
 if len(x)==3:
  print(x[0])
  print(x[1])
  print(x[2])
 else:
  print(x[0])
  print(x[1])
sys.exit()

no=-1
noun=[]
verb=[]
sem_feat=[[0]*25 for _ in range(60)]

with open("../data/sem_feat.txt","r") as inf:
 for line in inf:
  #in case we found concrete noun
   if line.startswith('F'):
    no+=1
    for word in cte_no:
     #give boundary to find exact word
     curr=re.findall('\\b'+word+'\\b',line)
     noun=noun+curr
   else:
    for word in verbs:
     if re.findall(word[0],line)!=[]:
      verb=re.findall(word[0],line)
      idx=find_verb(verb[0])
      temp=re.findall("\d.\d\d\d",line)
      contrib=float(temp[0])
      sem_feat[no][idx]=contrib
tp=np.dtype(float)
np.savetxt('sem_feat.txt', np.array(sem_feat,tp), fmt='%f')
tp=np.dtype('a')
np.savetxt('noun.txt', np.array(noun), fmt='%s')
