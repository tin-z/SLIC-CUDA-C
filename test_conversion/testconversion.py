#!/usr/bin/env python3
import os, subprocess, time

cmds = "nvcc -D DEBUG -O3 -arch=sm_20 test1.cu ../cimage/imagebmp.c ../converter.cu -o test1".split(" ")
subprocess.Popen(cmds, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

os.chdir("samples")
samples = os.listdir(".")
for out in [ x for x in samples if x.startswith("out_") ] :
  os.remove(out)

samples = [x for x in samples if not x.startswith("out_")]
failed = []

for sample in samples :
  p1 = subprocess.Popen(["../test1", sample], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
  p1.wait()
  


if failed == [] :
  print("[+] All the samples passed the byte a byte equals test")
else :
  print("[X] Some samples didn't pass the test..")
  print("   they are:{}".format(",".join(failed)))



