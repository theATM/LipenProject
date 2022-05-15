#!/bin/bash
#Rename Last Log File - to save current in it's place
#It must be run BEFORE a log changing python script (in pycharm you can set this up in run configuration)
#Check if logFile even exist
if test -f "ConsoleLogs/lastTrainLog.txt"; then
	#Get Modification date (to differentiate)
	DATE=$(date -r ConsoleLogs/lastTrainLog.txt "+_%m-%d-%Y_%H-%M-%S")
	#Create new unique name
	NEW_NAME="ConsoleLogs/oldTrainLog${DATE}.txt" 
	#Rename file NEW_NAME
	mv ConsoleLogs/lastTrainLog.txt $NEW_NAME
fi
