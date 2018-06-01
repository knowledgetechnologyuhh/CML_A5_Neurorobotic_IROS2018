


<b>A Neurorobotic Experiment for Crossmodal Conflict Resolution in Complex Environments</br> <br>


This repository helds the code used in the experiments of the following paper:

```sh
Parisi, G. I., Barros, P., Fu, D., Magg, S., Wu, H., Liu, X., Wermter, S. A Neurorobotic Experiment for Crossmodal Conflict Resolution in Complex Environments. Submitted to: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018. 
```

A Neurorobotic Experiment for Crossmodal Conflict Resolution in Complex Environments
Crossmodal conflict resolution is a crucial compo nent of robot sensorimotor coupling through interaction with the environment for swift and robust behaviour also in noisy conditions. In this paper, we propose a neurorobotic experiment in which an iCub robot exhibits human-like responses in a complex crossmodal environment. To better understand how humans deal with multisensory conflicts, we conducted a behavioural study exposing 33 subjects to congruent and incongruent dynamic audio-visual cues. In contrast to previous studies using simplified stimuli, we designed a scenario with four animated avatars and observed that the magnitude and extension of the visual bias are related to the semantics embedded in the scene, i.e., visual cues that are congruent with environmental statistics (moving lips and vocalization) induce a stronger bias. We propose a deep learning model that processes stereophonic sound, facial features, and body motion to trigger a discrete response resembling the collected behavioural data. After training, we exposed the iCub to the same experimental conditions as the human subjects, showing that the robot can replicate similar responses in real time. Our interdisciplinary work provides important insights into how crossmodal conflict resolution can be modelled in robots and introduces future research directions for the efficient combination of sensory drive with internally generated knowledge and expectations.
<br>
<b>Experimental Scenario</br> <br>

The Audio-Visual localization task consisted of the subjects having to select which avatar (out of the 4 avatars in the scene) they believe the auditory cue is coming from. The 4 avatars may move their lips and/or arm in temporal correspondence with an auditory cue. The latter consists of a vocalized combination of 3 syllables (all permutations without repetition composed of ”ha”, ”wa”, ”ba”). The duration of both visual and auditory stimuli is 1000 ms. The experiment comprised 5 AV conditions:
<br>
Baseline: Auditory cue and static avatars.<br>
Moving Lips: Auditory cue and one avatar with moving lips.<br>
Moving Arm: Auditory cue and one avatar with a moving arm.<br>
Moving Lips+Arm: Auditory cue and one avatar with moving lips and arm.<br>
Moving Lips–Arm: Auditory cue and one avatar with moving lips and another avatar with a moving arm.<br>

<b>Neurorobotic Model<br>

We propose a deep learning model processing both spatial and feature-based information in which low-level areas (such as the visual and auditory cortices) are predominantly unisensory, while neurons in higher-order areas encode multisensory representations. The proposed architecture comprises 3 input channels (audio, face, and body motion) and a hidden layer that computes a discrete behavioural response on the basis of the output of these unisensory channels.
<br>

<b>License
<br>
Our source code and corpus are distributed under the Creative Commons CC BY-NC-SA 3.0 DE license. If you use this corpus, you have to agree with the following items:
<br>
To cite our reference in any of your papers that make any use of the database and/or source code. The references are provided at the end of this page.<br>
To use the corpus and/or source code for research purpose only.<br>
To not provide the corpus and/or source code to any second parties.<br>
  
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

German I. Parisi - parisi@informatik.uni-hamburg.de






