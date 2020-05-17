import numpy as np

import serial

import time


waitTime = 0.1


# generate the waveform table
songFreq1=np.array([ #48 note

  294,294 ,294,294,294,294,196 , 
  294,294,294,330,262,294 ,
  196,294,294,294,294,196 ,
  294,294,294,370,294,294 ,
  370,370,370,370,392 ,196,
  370,330,330,370,370  ,370,
  294,294,370,370,330,330,
  330,330,330,330,294,294  
  ])


noteLength1 = np.array([

  0.5, 1.5, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 2, 1,

  1, 1, 1, 1, 3, 1,

  1, 1, 1, 1, 1, 1,
  
  1, 1, 1, 1, 1, 2  ])

songFreq2=np.array([  #42 note

  261, 261, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 261,

  392, 392, 349, 349, 330, 330, 294,

  392, 392, 349, 349, 330, 330, 294,

  261, 261, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 261])


noteLength2 = np.array([

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2,

  1, 1, 1, 1, 1, 1, 2])

songFreq3=np.array([  #26 note
  294,330,392 ,294,392,440 , 
  392,440,494 ,523,587 ,494,
  494,440,294,330,392  ,294,
  294,392,440 ,523,494 ,440,
  659 ,587])

noteLength3 = np.array([

  1, 1, 2, 1, 1, 2, 1,

  1, 2, 1, 2, 1, 1, 1,

  1, 1, 3, 1, 1, 1, 2,

  1, 2, 1, 2, 1])


# output formatter

formatter = lambda x: "%03.0f" % x

songList=np.array([songFreq2])
#, songFreq2, songFreq3])
lengthList=np.array([noteLength1])
#,noteLength2,noteLength3])
# send the waveform table to K66F

serdev = '/dev/ttyACM1'

s = serial.Serial(serdev)

print("Sending signal ...")

#print("It may take about %d seconds ..." % (int(signalLength * waitTime)))

for song in songList:
  for data in song:
    s.write(bytes(formatter(data), 'UTF-8'))
    print(bytes(formatter(data), 'UTF-8'))
    time.sleep(waitTime)

for length in lengthList:
  for data in length:
    s.write(bytes(formatter(data), 'UTF-8'))
    print(bytes(formatter(data), 'UTF-8'))
    time.sleep(waitTime)

s.close()

print("Signal sended")