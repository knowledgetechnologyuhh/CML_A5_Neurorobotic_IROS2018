<b>A Neurorobotic Experiment for Crossmodal Conflict Resolution in Complex Environments</br> <br>


This repository helds the code used in the experiments of the following paper:

```sh
Parisi, G. I., Barros, P., Fu, D., Magg, S., Wu, H., Liu, X., Wermter, S. A Neurorobotic Experiment for Crossmodal Conflict Resolution in Complex Environments. Submited to: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018. 
```

<b>Data Set </br>

During this project we performed a user study to evaluate how humans behave on crossmodal conflict solving. We replaced the humans by a iCub robot and collected audio-visual data, which was used to train a deep neural model based on the human responses. To have access to this dataset and the user responses, please contact us:

Pablo Barros - barros@informatik.uni-hamburg.de <br>
German I. Parisi - parisi@informatik.uni-hamburg.de<br>


<b>Requirements</br>

tensorflow-gpu <br>
keras<br>
matplotlib<br>
h5py<br>
opencv-python<br>
librosa<br>
imgaug<br>
dlib<br>


<b>Instructions</br>


This code has four different modes to run, one for each of the experiments (Audio Only, Lips Only, Arms Only and Crossmodal). Each of these specific codes read the specific datasets, build a deep neural model and train it. The crossmodal code also needs as input the user responses, as it uses the user responses to train the model instead of the unisensory stimuli.  <br>


<b> More information</br>

Project website: https://www.inf.uni-hamburg.de/en/inst/ab/wtm/research/cml.html

Pablo Barros - barros@informatik.uni-hamburg.de





