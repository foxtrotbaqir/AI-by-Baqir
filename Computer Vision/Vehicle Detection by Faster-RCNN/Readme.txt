the code runs for detection of 5 classes in a image sequence derived from video taken from KITTI benchmark.

Run the detector code.
aloing with it are given two local made CNNs you can use whichever fits your needs. just make sure to change the name while loading the network in code.

another variable which is loaded is 'scenelabels' that contains labeled ground truth data for video given.

as this is quite a cpu costly code so run it on a powerful system. after converted into a faster r-cnn it goes from a network
of KBs to a network of half a GB. So faster r-cnns need a powerful gpu to process the task. If any issue occur you can consult with me anytime. 

Thank you!