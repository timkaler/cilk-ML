import argparse
import os
import subprocess

from collections import OrderedDict

experiments = [
    {
        "name": "PARAD",
        "cores": [1,2,4,8,18],
        "binary": "run"
    },
    {
        "name": "WL",
        "cores": [1,2,4,8,18],
        "binary": "run_wl"
    },
    {
        "name": "PLOCKS",
        "cores": [1,2,4,8,18],
        "binary": "run_plocks"
    },
    {
        "name": "SERIAL",
        "cores": [1],
        "binary": "run_serial"
    }
]

algorithms = [
    {"name": "mlp1", "desc": "MLP 1 layer (800)"},
    {"name": "mlp2", "desc": "MLP 2 layer (400,100)"},
    {"name": "gcn1", "desc": "GCN Pubmed"},
    {"name": "gcn2", "desc": "GCN email-Eu-core"},
    {"name": "cnn1", "desc": "CNN lenet-5 (maxpool + ReLU)"},
    {"name": "cnn2", "desc": "CNN lenet-5 (average pool + tanh)"},
    {"name": "lstm1", "desc": "LSTM without added parallelism"},
    {"name": "lstm2", "desc": "LSTM with internal parallism"}
]

LOGS_DIR = "./logs"

################################################################################

def taskset_string(cores):
    return "taskset -c 0-" + str(cores-1)

def shell_get_output(command):
    process = subprocess.Popen(command, shell=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if len(err) > 0:
        raise Exception("command: {}\noutput: {}\nerr: {}", command, output, err)
    return output


# Run a single experiment and write its output to a file
def run_single_experiment(alg_num, name, binary, cores):
    output_file = "{}/{}_{}_cores_{}".format(
            LOGS_DIR, name, algorithms[alg_num]['name'], cores)
    command = "{} ./setup.sh ./{} -a {}".format(
            taskset_string(cores), binary, alg_num)
    output = shell_get_output(command)
    open(output_file, "wb").write(output)

def run_experiments():
    # Ensure that all necessary file dependencies already exist
    if os.path.exists(LOGS_DIR):
        raise Exception("The folder " + LOGS_DIR + " already exists")
    os.makedirs(LOGS_DIR)
    for experiment in experiments:
        if not os.path.exists("./{}".format(experiment['binary'])):
            raise Exception("Cannot find ./{}".format(experiment['binary']))
    # Run all the experiments sequentially 
    for experiment in experiments:
        for cores in experiment['cores']:
            for alg_num in range(len(algorithms)):
                run_single_experiment(alg_num, experiment['name'], experiment['binary'], cores)

# Parses the forward, reverse, and forward+reverse runtimes
def parse_single_result(alg_num, name, binary, cores):
    output_file = "{}/{}_{}_cores_{}".format(
            LOGS_DIR, name, algorithms[alg_num]['name'], cores)
    lines = open(output_file, "r").readlines()
    data = {}
    for l in lines:
        if l.strip().startswith("Forward pass"):
            data["FORWARD"] = l.strip().split(" ")[-1]
        elif l.strip().startswith("Reverse pass"):
            data["REVERSE"] = l.strip().split(" ")[-1]
        elif l.strip().startswith("Forward+Reverse pass"):
            data["FORWARD+REVERSE"] = l.strip().split(" ")[-1]
    return data

# Read output results and generate a nice output table
def parse_results():
    output = OrderedDict()
    for experiment in experiments:
        output[experiment['name']] = OrderedDict()
        for cores in experiment['cores']:
            output[experiment['name']][cores] = OrderedDict()
            for alg_num in range(len(algorithms)):
                data = parse_single_result(alg_num, experiment['name'], experiment['binary'], cores)
                output[experiment['name']][cores][algorithms[alg_num]['name']] = data
    print(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', help="Run experiments", action='store_true')
    parser.add_argument('--parse', help="Parse results", action='store_true')
    args = parser.parse_args()

    if args.run:
        run_experiments()
    if args.parse:
        parse_results()
    if not args.run and not args.parse:
        raise Exception("Must specify --run, --parse, or both")
