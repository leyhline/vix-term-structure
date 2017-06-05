#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
	echo Usage: "$0" '<source directory> <target framerate>'
	exit
fi

sourcedir=$1 
framerate=$2

if [ -d "$sourcedir" ]; then
	cd "$sourcedir"
	ffmpeg -y -framerate "$framerate" -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "out_${framerate}.mp4"
else
	echo "$sourcedir" does not exist.
fi
