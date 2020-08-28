#!/usr/bin/python3

import os
import json

with open("./gridsearch.json") as json_combi:
    data = json.load(json_combi)
    command = data["command"]
    combinations = data["combinations"]
    for i in range(len(combinations) - 1):
        os.system(command + " -g {}".format(i))
