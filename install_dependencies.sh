#!/bin/bash
#virtualenv -q -p /usr/bin/python3.6 $1

# Get pip3 if not already on system
sudo apt-get install python3-pip

# Get test-specific dependencies
pip3 install -r requirements.txt

sudo apt-get install python3-tk




