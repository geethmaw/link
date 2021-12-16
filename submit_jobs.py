#!/usr/bin/python
# Example PBS cluster job submission in Python

from subprocess import Popen, PIPE
import time
import sys

# If you want to be emailed by the system, include these in job_string:
#PBS -M your_email@address
#PBS -m abe  # (a = abort, b = begin, e = end)

# Loop over your jobs
for i in range(1, 3):

    # Open a pipe to the qsub command.
    proc = Popen('qsub', shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)

    # Customize your options here
    job_name = "my_job_%d" % i
    walltime = "01:00:00"
    processors = "nodes=1:ncpus=36"
    command = "bVSrr.py %d" % i

    job_string = """
    #!/bin/bash
    #PBS -A WYOM0128
    #PBS -N %s
    #PBS -l walltime=%s
    #PBS -l %s
    #PBS -M wgeethma@uwyo.edu
    #PBS -m abe
    #PBS -q regular
    python %s""" % (job_name, walltime, processors, command)

    # Send job_string to qsub
    if (sys.version_info > (3, 0)):
        proc.stdin.write(job_string.encode('utf-8'))
        print('k')
    else:
        proc.stdin.write(job_string)
        print('kk')
    out, err = proc.communicate()

    # Print your job and the system response to the screen as it's submitted
    print(job_string)
    print(out)

    time.sleep(0.1)
