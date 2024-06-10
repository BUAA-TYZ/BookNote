#! /f/anaconda3/python
import subprocess

out = subprocess.check_output(["ls", "-a"]).decode("utf-8")
print(out)
