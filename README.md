# Interoception Training with HBD

0.Summary:
-----------------
Using webcam to detect heartbeat(pulse) through fingertip and conducting heartbeat discrimination training(Including confidence reporting feature)

1.How this Code work
-----------------
This interoception training program use heartbeat discrimination(HBD) task as the training method. After each HBD trial, the participants need to report their decision confidence and will be given the right answer as a feedback to improve the participants' interoception. At the same time, conductor can also change this code to not giving right answer, in this scenario, this program will be switched to a interoception accuracy(IcA) test. 

When using this program, the participants need to put their finger on the webcam. The program will get frames from the webcam. As the heart contrasts and expands, the blood being injected to fingertips regularly, the transparency of the fingertips will also change regularly. Consequently, the brightness of the frames obtained from the webcam will change regularly. We can then depict the wave graph of the brightness of the real-time frames. Each peak on this wave graph corresponds a heartbeat(so the brightness will decrease after the heartbeat (peak) ). In fact, the blood need approximate 250ms to be transferred from the heart to fingertip, so the time of the peak is 250ms after the real heartbeat of the participant. This 250ms delay has been taken account into this program[1].

Since we have already obtained the heartbeat, then we can show this heartbeat in the form of a red dot flash on the screen to the participants. The participants can then 'see' their own heartbeats. Also, we can add some delay on the red dot flash. In this scenario, if the participants have good interoception, they can tell the flash displayed on the screen is not their own heartbeat(asynchronized with their own heartbeat). In all, for each trial, we have two choices, to display a either delayed or not delayed red dot flash, and then ask the participants whether the displayed flash is synchronized or asynchronized with their own heartbeats. We repeat this kind of  trial for N times, and then we can train/test the participants' heartbeat discrimination ability(Interoception ability).


2.Requirements:
---------------

- Python v2.7 or v3.5+
- OpenCV v2+
- Numpy, Scipy

3.Usage:
------------
### 3.1 Run
- run interoception_training.py to start the program

```
python interoception_training.py
```
### 3.2 Procedure
Run program:
- If detected brightness too low, warning
- Presss "C" to change camera
- Put your finger on
- Watch the oscillogram until it produce regular wave shape
- Press R to enter Reflection Mode: Try to feel your heart beat in 1 min
- When you are ready, press "T" to start training
	- Onece start quiz, could not press "C" button anymore
- Training mode
	- Trial 1:
		- Displaying 10s red dot flash and wave
		- Displaying question: Asyn or Syn?
			- Wiat for input of answer(Press Y or N)
		- Displaying question: what about your confidence?
			- Wiat for input (Press 1, 2, or 3)
		- Display right answer/feedback
		- Press T to continue to next trial
	- Trial 2:
		- Repeat above
	- Trial N:
		- Repeat above
	- Save training data


### 3.3 Important parameters
In lib/frame_processor.py, you can modify following parameters according to your needs:
```
self.reflection_time=60    #the total time(seconds) required in the reflection mode
self.trial_num = 20        #the number of trials in training mode
self.trial_duration = 10   #how many seconds the participants was given to judge before they need to answer the questions in each trial
```

### 3.4 About the webcam
If the computer is connected with more than one camera, then the user may presss 'C' to change the camera at the beginning of the program. In particular, if this program is running on a Mac, and the user have iPhone as well, then the  camera of iPhone will be automatically connected to Mac if the Mac and iPhone meet the requirements specificed in this [page](https://support.apple.com/guide/mac-help/use-iphone-as-a-webcam-mchl77879b8a/mac) 

4.Data Structure and Saving
----------
This program will automatically record all the brightness value of all the frames while the program is running. These value could be used to calculate the participants' HRV during the procedure. This program will also record the answer of the participants made in all trials and their confidence levels. All the data will be saved in the path: **'[home path]/Interoception_training_data/'** in the form of .npz([numpy.savez](https://numpy.org/doc/stable/reference/generated/numpy.savez.html))
### Data Structure:
![](https://drive.google.com/uc?id=1aCsEAero2sLnSIc2LJKUCrHD8GdGSIXB)


5.Acknowledge and References
----------
The procedure of this program is inspired by reviewing recent work:

[1] Quadt, L., Garfinkel, S. N., Mulcahy, J. S., Larsson, D. E., Silva, M., Jones, A. M., ... & Critchley, H. D. (2021). Interoceptive training to target anxiety in autistic adults (ADIE): A single-center, superiority randomized controlled trial. EClinicalMedicine, 39, 101042. ([Paper](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(21)00322-9/fulltext))

The code is inspired and partly based on thearn's work on [webcam pulse detection](https://github.com/thearn/webcam-pulse-detector). Licensed under the Apache License.
