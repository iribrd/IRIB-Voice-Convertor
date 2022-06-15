import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np



sig, fs = librosa.load('chunk0_re.wav')  
sig2 = open('chunk0_re.npy')   
 
# make pictures name 
save_path1 = 'chunk0_re_new.jpg'
save_path2 = 'chunk0_re_old.jpg'

pylab.axis('off') # no axis
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
S = librosa.feature.melspectrogram(y=sig, sr=fs)

np.save('1.npy',S.astype(np.float32), allow_pickle=False) 

#librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
librosa.display.specshow(librosa.power_to_db(sig2, ref=np.max))

pylab.savefig(save_path2, bbox_inches=None, pad_inches=0)
pylab.close()



# import matplotlib.pyplot as plt
# import librosa

# plt.figure()
# librosa.display.specshow(spec, x_axis='linear')
