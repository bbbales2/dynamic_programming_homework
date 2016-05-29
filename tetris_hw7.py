#%%
import os
import pyximport
pyximport.install()

os.chdir('/home/bbales2/Documents/classes/ece271/')
import tetris

tetris.run(10, 10, 0.1, 0.99, 0.95)
