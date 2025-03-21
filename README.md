How to run it:
1. connect to vpn.utah.edu
2. ssh notchpeak.chpc.utah.edu
3. scp -r /Users/u1520797/Downloads/assignment1package u1520797@notchpeak.chpc.utah.edu:~
4. in the slur file change the partition and account by this info:
<img width="870" alt="image" src="https://github.com/user-attachments/assets/9e8e2999-e306-42c7-945f-0487b55b8ff4" />

5. sbatch slurm.sh
6. squeue -u u1520797
7. in the slurm.sh file go to the logs directory and see the output in the log files and the graphs are generated in png format
