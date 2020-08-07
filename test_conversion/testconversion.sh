#!/bin/bash

SAMPLE="./samples"

ls $SAMPLE | while read x;
do
  nvprof ./test1 "$SAMPLE/$x"  2>&1 | grep -e "convertRGB2Lab" -e "convertLab2RGB"
  echo
done

