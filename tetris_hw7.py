#%%
import os
import pyximport
pyximport.install()

os.chdir('/home/bbales2/Documents/classes/ece271/')
import tetris
import time

tmp = time.time()
tetris.run(10, 10, 0.1, 0.99, 0.95)
print "Total: ", time.time() - tmp

print tetris.gns
