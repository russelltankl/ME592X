% Assignment 1
% 2.1
clear;close all;clc

% Import image
I = imread('sudoku-original.png');

% display image
imshow(I);

% Convert to grayscale
I = rgb2gray(I);
figure
imshow(I)
whos I

% Display pixel intensities histogram (contrast of range 0~ 255)
figure
imhist(I)