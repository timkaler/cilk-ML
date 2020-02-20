import subprocess
import sys
import random
import os

def shellGetOutput(str) :
  process = subprocess.Popen(str,shell=True,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  output, err = process.communicate()

  if (len(err) > 0):
      raise NameError(str+"\n"+output+err)
  return output



names = ["mlp_1_layer", "gcn_pubmed_1_layer", "mlp_2_layer", "cnn_lenet5", "cnn_lenet5_tanh", "gcn_enron"]


FLOAT_FORMAT_STRING = "%.2f"

def run_experiment(alg, binary, cores):
  output_file = "logs/raw_"+binary+"_alg_"+str(names[alg])+"_"+str(cores)
  result_file = "logs/result_"+binary+"_alg_"+str(names[alg])+"_"+str(cores)
  taskset_string = " taskset -c 0 "
  if cores > 1:
    taskset_string = " taskset -c 0-"+str(cores-1)+" "
  cmd_string = taskset_string + " ./setup.sh ./"+binary+" -a " + str(alg)

  output = shellGetOutput(cmd_string)

  open(output_file, "w").write(output)


def parse_experiment(alg, binary, cores):
  output_file = "logs/raw_"+binary+"_alg_"+str(names[alg])+"_"+str(cores)
  lines = open(output_file, "r").readlines()
  data = [str(binary), names[alg], str(cores)]
  for l in lines:
    if l.strip().startswith("Forward pass") or l.strip().startswith("Reverse pass") or l.strip().startswith("Forward+Reverse pass"):
      data.append(l.strip().split(" ")[-1])

  BIN = "None"
  if binary.startswith("run_parallel"):
    BIN = "PARAD"
  if binary.startswith("run_serial"):
    BIN = "SERIAL"
  if binary.startswith("run_plocks"):
    BIN = "LOCKS"

  keyname = BIN + "_TIME_" + str(cores) + "_" + names[alg]

  return [(keyname + "_FORWARD", data[3]), (keyname + "_REVERSE", data[4]), (keyname + "_FORWARDREVERSE", data[5])]

  #print "\t".join(data)



algorithms = [0,1,2,3,4,5]
binaries = ["run_parallel", "run_serial", "run_plocks"]
cores = [[18,8,4,2,1], [1], [18,8,4,2,1]]




if 0:
  for a in algorithms:
    #if a != 5:
    #  continue
    for i in range(0,len(binaries)):
      b = binaries[i]
      for c in cores[i]:
        run_experiment(a,b,c)
else:
  d = dict()
  for a in algorithms:
    for i in range(0,len(binaries)):
      b = binaries[i]
      for c in cores[i]:
        items = parse_experiment(a,b,c)
        for x in items:
          d[x[0]] = x[1]
  print d










#output = shellGetOutput("taskset -c 0-17 ./setup.sh ./run_parallel -a 0")



#for l in output.splitlines():
#  if l.strip().startswith("Forward pass"):
#    print l
#  if l.strip().startswith("Reverse pass"):
#    print l
#  if l.strip().startswith("Forward+Reverse pass"):
#   print l
