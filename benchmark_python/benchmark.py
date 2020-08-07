#!/usr/bin/env python3
import os, sys, re, subprocess
import time

## Config
K_sizes = [300, 600, 1000]
M_param = 40

to_test = {'gmem':('slic.cu', {x:0 for x in K_sizes}) ,\
           'smem':('slic_smem.cu', {x:0 for x in K_sizes}) ,\
           'sync':('slic_sync.cu', {x:0 for x in K_sizes}) ,\
           'bitm':('slic_bitmap.cu', {x:0 for x in K_sizes})  }

folder_samples = "test_images"
samples = [ "{}/{}".format(folder_samples, sample) for sample in os.listdir(folder_samples) if sample.endswith(".bmp") ]

function_pattern = ['initClusters(', 'assignment(', 'update(']



def test_complexity() :
  ## Config
  global K_sizes, M_param, to_test, folder_samples, samples, function_pattern

  ## Benchmark
  for arch,prop in to_test.items() :
    print("Doing arch:{}".format(arch))

    file_arch = prop[0]
    subprocess.getoutput("nvcc -D DEBUG -O3 -arch=sm_20 {} cimage/imagebmp.c converter.cu -o slic".format(file_arch))

    for k in K_sizes :
      cc = 0.

      for sample in samples :
        
        time.sleep(1)
        rets = subprocess.getoutput("nvprof ./slic {} {} {} -F".format(sample, k, M_param))
        rets = [ x for x in rets.split("\n") if len([ y for y in function_pattern if y in x ]) > 0 ]
        assert len(rets) == len(function_pattern)
       
        cc2 = 0. 
        for elem in rets :
          #print(elem)
          regex="^[ ][ ]*\d{1,5}\.\d{1,5}%[ ][ ]*(\d{1,5}.\d{1,5})(ms|us|s)[ ][ ]+.*"
          rets_tmp = re.search(regex, elem)
          sec_tmp = float(rets_tmp.group(1))
          sec_typ = rets_tmp.group(2)

          if sec_typ == "us" :
            sec_tmp = sec_tmp * 0.001
          elif sec_typ == "s" :
            sec_tmp = sec_tmp * 1000

          cc2 += sec_tmp

        cc += cc2
        #break

      cc /= len(samples)
      prop[1][k] = cc
      print("  done K:{} result:{:.2f}".format(k, prop[1][k]))
      
      #break
    #break

  ## Print results
  print("\n                       K:300       K:600      K:1000")
  for arch,prop in to_test.items() :
    print("Result arch:{}      ".format(arch) + "{:.2f}     {:.2f}      {:.2f}".format(*[(prop[1][k]) for k in K_sizes ]))

 

def test_occupancy(fixed_k=1000):
  global K_sizes, M_param, to_test, folder_samples, samples, function_pattern
  samples = samples[:10]
  occupancy = { k:{ x:0. for x in function_pattern } for k in list(to_test.keys()) }

  for arch,prop in to_test.items() :
    file_arch = prop[0]
    subprocess.getoutput("nvcc -D DEBUG -O3 -arch=sm_20 {} cimage/imagebmp.c converter.cu -o slic".format(file_arch))

    for sample in samples :
      time.sleep(1)
 
      rets = subprocess.getoutput("nvprof --metrics achieved_occupancy ./slic {} {} {} -F".format(sample, fixed_k, M_param))
      rets = rets.split("Metric result:")[1].strip().split("\n")
        
      for idx,x in enumerate(rets) :
        if x.startswith("Device \"") :
          cc_tmp = idx
        
      rets = rets[cc_tmp+1:]

      for index in list(range(0, len(rets)-2, 2)) :
        for f in function_pattern : 
          if f in rets[index] :
            occupancy[arch][f] += float( rets[index+1].split("Achieved Occupancy")[1].split(" ")[-1] )
      
      #break
    #break

  for arch,prop in occupancy.items() :
    for f in function_pattern:
      occupancy[arch][f] /= len(samples)

  ## Print results
  print("\n                            "+"       ".join([ x[:-1] for x in function_pattern]))
  for arch,prop in occupancy.items() :
    print("Occupancy arch:{}             ".format(arch) + "{:.2f}              {:.2f}           {:.2f}".format(*[prop[f] for f in function_pattern ]))



def test_branch_efficiency(fixed_k=1000):
  global K_sizes, M_param, to_test, folder_samples, samples, function_pattern
  samples = samples[:10]
  occupancy = { k:{ x:0. for x in function_pattern } for k in list(to_test.keys()) }

  for arch,prop in to_test.items() :
    file_arch = prop[0]
    subprocess.getoutput("nvcc -D DEBUG -O3 -arch=sm_20 {} cimage/imagebmp.c converter.cu -o slic".format(file_arch))

    for sample in samples :
      time.sleep(1)
 
      rets = subprocess.getoutput("nvprof --metrics branch_efficiency ./slic {} {} {} -F".format(sample, fixed_k, M_param))
      rets = rets.split("Metric result:")[1].strip().split("\n")
        
      for idx,x in enumerate(rets) :
        if x.startswith("Device \"") :
          cc_tmp = idx
        
      rets = rets[cc_tmp+1:]

      for index in list(range(0, len(rets)-2, 2)) :
        for f in function_pattern : 
          if f in rets[index] :
            occupancy[arch][f] += float( rets[index+1].split("Branch Efficiency")[1].split(" ")[-1][:-1] )
      
      #break
    #break

  for arch,prop in occupancy.items() :
    for f in function_pattern:
      occupancy[arch][f] /= len(samples)

  ## Print results
  print("\n                                         "+"       ".join([ x[:-1] for x in function_pattern]))
  for arch,prop in occupancy.items() :
    print("Branch efficiency(%) arch:{}             ".format(arch) + "{:.2f}              {:.2f}         {:.2f}".format(*[prop[f] for f in function_pattern ]))



def test_throughput(fixed_k=1000):
  global K_sizes, M_param, to_test, folder_samples, samples, function_pattern
  samples = samples[:10]
  efficiency = { k:{ x:[0.,0.] for x in function_pattern } for k in list(to_test.keys()) }

  for arch,prop in to_test.items() :
    file_arch = prop[0]
    subprocess.getoutput("nvcc -D DEBUG -O3 -arch=sm_20 {} cimage/imagebmp.c converter.cu -o slic".format(file_arch))

    for sample in samples :
      time.sleep(1)
 
      rets = subprocess.getoutput("nvprof --metrics gld_efficiency,gst_efficiency ./slic {} {} {} -F".format(sample, fixed_k, M_param))
      rets = rets.split("Metric result:")[1].strip().split("\n")
        
      for idx,x in enumerate(rets) :
        if x.startswith("Device \"") :
          cc_tmp = idx
        
      rets = rets[cc_tmp+1:]

      for index in list(range(0, len(rets)-3, 3)) :
        for f in function_pattern : 
          if f in rets[index] :
            efficiency[arch][f][0] += float( rets[index+1].split("Load Efficiency")[1].split(" ")[-1][:-1] )
            efficiency[arch][f][1] += float( rets[index+2].split("Store Efficiency")[1].split(" ")[-1][:-1] )
      
      #break
    #break

  for arch,prop in efficiency.items() :
    for f in function_pattern:
      efficiency[arch][f][0] /= len(samples)
      efficiency[arch][f][1] /= len(samples)

  ## Print results
  print("\n                                         "+"       ".join([ x[:-1] for x in function_pattern]))
  for arch,prop in efficiency.items() :
    prop0 = [ prop[f][0] for f in function_pattern ]
    prop1 = [ prop[f][1] for f in function_pattern ]
    print("Load,Store efficiency(%) arch:{}             ".format(arch))
    print("                                         "+"{:.2f}              {:.2f}         {:.2f}".format(*prop0))
    print("                                         "+"{:.2f}              {:.2f}         {:.2f}".format(*prop1))


 

if len(sys.argv) < 2 :
  print("Wrong, usage: python3 {} <complexity|occupancy|branch|throughput>".format(sys.argv[0]))
  sys.exit(-1)


if sys.argv[1] == "complexity" :
  test_complexity()

elif sys.argv[1] == "occupancy" :
  test_occupancy(fixed_k=1000)

elif sys.argv[1] == "branch" :
  test_branch_efficiency()

else :
  test_throughput()



