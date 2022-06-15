import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np

normalized_image = np.load('chunk0_re.npy')

#plt.imshow(normalized_image)

save_path2 = 'chunk0_re_old.jpg'
librosa.display.specshow(normalized_image)
pylab.savefig(save_path2, bbox_inches=None, pad_inches=0)
pylab.close()



# import matplotlib.pyplot as plt
# import librosa

# plt.figure()
# librosa.display.specshow(spec, x_axis='linear')
