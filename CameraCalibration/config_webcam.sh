#!/bin/sh

# configure for low exposure needed for retroreflective tape
sudo uvcdynctrl --set="Exposure, Auto" 3
sudo uvcdynctrl --set="Exposure, Auto Priority" 1

sudo uvcdynctrl --set="White Balance Temperature, Auto" 1

sudo uvcdynctrl --set="Brightness" 128
sudo uvcdynctrl --set="Contrast" 32
sudo uvcdynctrl --set="Saturation" 34

sudo uvcdynctrl --set="Power Line Frequency" 1



