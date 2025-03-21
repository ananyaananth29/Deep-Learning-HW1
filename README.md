How to run it:
1. connect to vpn.utah.edu
2. ssh notchpeak.chpc.utah.edu
3. scp -r /Users/u1520797/Downloads/assignment1package u1520797@notchpeak.chpc.utah.edu:~
4. sbatch slurm.sh
5. squeue -u u1520797
6. in the slurm.sh file go to the logs directory and see the output in the log files and the graphs are generated in png format
