import gobject
import sys
from annpkg.model import AnnPkg
import gui
import json
import numpy as np

gobject.threads_init()

training_file = sys.argv[1]
classification = sys.argv[2]
target_file = sys.argv[3]

mapping = {}
for classifier in classification.split("#"):
    name,cls = classifier.split(':')
    mapping[name] = cls.split(',')

t = AnnPkg.load(training_file)

mp = {}
for dat in t.movement_data():
    if dat[0] in mp:
        mp[dat[0]].append(np.array(dat[1:]))
    else:
        mp[dat[0]] = [np.array(dat[1:])]

def done(fis):
    with open(target_file,'w') as h:
        json.dump(fis.to_json(),h,indent=True)

thread = gui.TrainingProgress(mapping,mp,done=done)
thread.start()
thread.join()
