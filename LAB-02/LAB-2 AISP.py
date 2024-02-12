#!/usr/bin/env python
# coding: utf-8

# In[23]:


#A1
import librosa
import matplotlib.pyplot as plt
y, rs = librosa.load('Ai.wav')
plt.figure(figsize=(10, 5))
librosa.display.waveshow(y, sr=rs,color='red')
plt.title('Original Signal')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.grid(True)
plt.tight_layout()
plt.show()
print("Playing Original Signal:")
ipd.Audio(y, rate=sr)



# In[21]:


#A1
y, rs = librosa.load('Ai.wav')
derivative_1 = np.diff(y)
derivative_1 /= np.max(np.abs(derivative_1))

plt.figure(figsize=(10, 5))
librosa.display.waveshow(derivative_1, sr=rs,color='black')
plt.title('First derivative of Original Signal')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.grid(True)
plt.tight_layout()
plt.show()

print("Playing First Derivative Signal:")
ipd.Audio(derivative_1, rate=rs)


# In[24]:


#A2
zero_crossing = np.where(np.diff(np.sign(derivative_1)))[0]
diff = np.diff(zero_crossing)
threshold = 1000
speech_regions = diff[diff > threshold]
silence_regions = diff[diff <= threshold]

avg_length_speech = np.mean(speech_regions)
avg_length_silence = np.mean(silence_regions)

print("Average length between consecutive zero crossings in speech regions:", avg_length_speech)
print("Average length between consecutive zero crossings in silence regions:", avg_length_silence)

plt.figure(figsize=(10, 5))
plt.plot(diff, label='All regions',color = 'purple')
plt.plot(np.arange(len(speech_regions)), speech_regions, 'ro', label='Speech regions',color = 'red')
plt.plot(np.arange(len(speech_regions), len(speech_regions) + len(silence_regions)), silence_regions, 'bo', label='Silence regions',color = 'blue')
plt.title('Pattern of Zero Crossings')
plt.xlabel('Zero Crossing Difference')
plt.ylabel('Difference between Consecutive Zero Crossings')
plt.legend()
plt.show()

print("Pattern of Zero Crossings:")
print("All regions:", diff)
print("Speech regions:", speech_regions)
print("Silence regions:", silence_regions)


# In[31]:


#A3
word_files_mine = ['apple.mp3', 'ball.mp3', 'cat.mp3', 'dog.mp3', 'elephant.mp3']
word_files_team_mate = ['apple_s.wav', 'ball_s.wav', 'cat_s.wav', 'dog_s.wav', 'elephant_s.wav']
words = ['Apple', 'Ball', 'Cat', 'Dog', 'Elephant']
word_lengths_mine = []
word_lengths_teammate = []

for word_file in word_files_mine:
    signal, sr = librosa.load(word_file, sr=None)
    length_seconds = len(signal) / sr
    word_lengths_mine.append(length_seconds)

for word_file in word_files_team_mate:
    signal, sr = librosa.load(word_file, sr=None)
    length_seconds = len(signal) / sr
    word_lengths_teammate.append(length_seconds)

print("Lengths of words of Abhi:", word_lengths_mine)
print("Lengths of words of Sangeethi:", word_lengths_teammate)

bar_width = 0.35
index = np.arange(len(words))
plt.figure(figsize=(12, 6))
plt.bar(index - bar_width/2, word_lengths_mine, bar_width, label='My Words', color='yellow')
plt.bar(index + bar_width/2, word_lengths_teammate, bar_width, label="Teammate's Words", color='red')
plt.xlabel('Words')
plt.ylabel('Length (seconds)')
plt.title('Comparison of Spoken Words Length')
plt.xticks(index, words)
plt.legend()

plt.show()


# In[30]:


#A4
statement, sr1 = librosa.load('neutral.mp3')
question, sr2 = librosa.load('questioning.mp3')
plt.figure(figsize=(10, 5))
librosa.display.waveshow(statement, sr=sr,color='blue')
plt.title('STATEMENT')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

question, sr = librosa.load('questioning.mp3')
plt.figure(figsize=(10, 5))
librosa.display.waveshow(statement, sr=sr,color='blue')
plt.title('QUESTION')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




