#!/bin/bash

# show the difference between two binary files with colors
colordiff -y <(xxd file1.bin) <(xxd file2.bin) | less