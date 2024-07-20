# Automated License Plate Recognition System

## Description
The goal of this project is to develop an Automated License Plate Recognition (ALPR) system which returns the license plate numbers of cars that pass by a toll point camera.

## Table of Contents
- [Installation](#installation)
- [Features](#features)

## Installation
Create a virtual environment and install requirements from  `requirements.txt` as follows:
`pip install -r /path/to/requirements.txt`

Alternatively, this project is also hosted as a microservice on Dockerhub.

## Features
This project extracts frames from streaming video input from a traffic camera, performs object detection via YOLOv3 to isolate the license plates from the cars in the video, and then extracts the license plate characters using EasyOCR. 
