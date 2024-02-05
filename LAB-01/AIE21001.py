#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install librosa')


# In[8]:


import librosa
y, sr = librosa.load('Ai.wav')


# In[9]:


librosa.display.waveshow(y)


# In[25]:


print("Signal length:",len(y),"samples")
print("Range of Magnitude:",np.min(y),"to", np.max(y))
print("Rate of Sampling:",sr, "Hz")


# In[10]:


get_ipython().system('pip install pydub')


# In[49]:


# A3 
import IPython.display as ipd
start = int(0.523 * sr)
end = int(1.3 * sr)
segment = y[start:end]
librosa.display.waveshow(segment, color = 'red')
ipd.Audio(segment, rate=sr)


# In[50]:


# A3
import IPython.display as ipd
start = int(1.3 * sr)
end = int(1.6 * sr)
segment = y[start:end]
librosa.display.waveshow(segment, color = 'blue')
ipd.Audio(segment, rate=sr)


# In[51]:


# A3
import IPython.display as ipd
start = int(1.6 * sr)
end = int(2.0 * sr)
segment = y[start:end]
librosa.display.waveshow(segment, color = 'green')
ipd.Audio(segment, rate=sr)


# In[52]:


# A3
import IPython.display as ipd
start = int(2.0 * sr)
end = int(2.8 * sr)
segment = y[start:end]
librosa.display.waveshow(segment, color = 'yellow')
ipd.Audio(segment, rate=sr)


# In[54]:


# A3 playing the noise of the signal 
import IPython.display as ipd
start = int(0.0 * sr)
end = int(0.5 * sr)
segment = y[start:end]
librosa.display.waveshow(segment, color = 'black')
ipd.Audio(segment, rate=sr)


# In[60]:


# A2
print("The words in the speech have a higher magnitude when compared to the noise signal.")
print("By observing the noise signal, the highest magnitude is approximately 0.3, while in the processed signal, it is more than 0.75.")


# In[67]:


# A4
import IPython.display as ipd
start1 = int(0.0 * sr)
end1 = int(0.5 * sr)
start2 = int(0.53 * sr)
end2 = int(1.3 * sr)
start3 = int(1.3 * sr)
end3 = int(2.0 * sr)
start4 = int(2.0 * sr)
end4 = int(2.8 * sr)

segment1 = y[start1:end1]
segment2 = y[start2:end2]
segment3 = y[start3:end3]
segment4 = y[start4:end4]

combined_segment = np.concatenate([segment1, segment2, segment3, segment4])
librosa.display.waveshow(combined_segment, color='pink')
ipd.Audio(combined_segment, rate=sr)





# In[ ]:




