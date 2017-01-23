#!/bin/sh

# configure for low exposure needed for retroreflective tape
sudo uvcdynctrl --set="Exposure, Auto" 1
sudo uvcdynctrl --set="Exposure, Auto Priority" 0
sudo uvcdynctrl --set="Exposure (Absolute)" 100

sudo uvcdynctrl --set="White Balance Temperature, Auto" 0
sudo uvcdynctrl --set="White Balance Temperature" 4350

sudo uvcdynctrl --set="Brightness" 140
sudo uvcdynctrl --set="Contrast" 255
sudo uvcdynctrl --set="Saturation" 50

sudo uvcdynctrl --set="Power Line Frequency" 1



