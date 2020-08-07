#!/usr/bin/env python3
import subprocess, sys, os

outdir_root = "out_test_images"
indir ="test_images"
m = "5"
k_list = ["200", "300", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "2000"] #, "2400", "2800", "3200", "3600", "4000", "4600", "5200"]

bmps = [ x for x in os.listdir(indir) if x.endswith(".bmp") ]


for k in k_list :

  outdir = "{}/{}".format(outdir_root, k)
  try :
    os.makedirs(outdir)
  except FileExistsError as ex :
    pass

  for image in bmps:
    p1 = subprocess.Popen(["./slic", "{}/{}".format(indir, image), k, m, "-I", "25", "-T", "-o", "{}/{}".format(outdir, image.split(".")[0]+".csv")], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    p1.wait()
    print("Done: {}/{}".format(outdir, image))

  print("Done folder: {}".format(outdir))

print("## Done all .. exit")

