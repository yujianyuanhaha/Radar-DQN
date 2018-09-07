#!/bin/bash

# Example qsub script for Cascades

# NOTE: You will need to edit the Walltime, Resource Request, Queue, and Module lines
# to suit the requirements of your job. You will also, of course have to replace the example job
# commands below with those that run your job.
 
#### Resource Request: ####
# Cascades has the following hardware:
#   a. 190 32-core, 128 GB Intel Broadwell nodes
#   b.   4 32-core, 512 GB Intel Broadwell nodes with 2 Nvidia K80 GPU
#   c.   2 72-core,   3 TB Intel Broadwell nodes
#
# Resources can be requested by specifying the number of nodes, cores, memory, GPUs, etc
# Examples:
#   Request 2 nodes with 32 cores each
#   #PBS -l nodes=1:ppn=32
#   Request 4 cores (on any number of nodes)
#   #PBS -l procs=4
#   Request 12 cores with 20gb memory per core
#   #PBS -l procs=12,pmem=20gb
#   Request 2 nodes with 32 cores each and 20gb memory per core (will give two 512gb nodes)
#   #PBS -l nodes=2:ppn=32,pmem=20gb
#   Request 2 nodes with 32 cores per node and 1 gpu per node
#   #PBS -l nodes=2:ppn=32:gpus=1
#   Request 2 cores with 1 gpu each
#   #PBS -l procs=2,gpus=1
PBS -l nodes=1:ppn=32:gpus=1

#### Walltime ####
# Set the walltime, which is the maximum time your job can run in HH:MM:SS
# Note that if your job exceeds the walltime estimated during submission, the scheduler
# will kill it. So it is important to be conservative (i.e., to err on the high side)
# with the walltime that you include in your submission script. 
PBS -l walltime=18:00:00

#### Queue ####
# Queue name. Cascades has four queues:
#   normal_q   for production jobs on all Haswell nodes
#   largemem_q for jobs on the two 3TB, 60-core Ivy Bridge servers
#   dev_q      for development/debugging jobs. These jobs must be short but can be large.
#   vis_q      for visualization jobs. These jobs must be both short and small.
PBS -q normal_q

#### Account ####
# This determines which allocation this job's CPU hours are billed to.
# Replace "youraccount" below with the name of your allocation account.
# If you are a student, you will need to get this from your advisor.
# For more on allocations, go here: http://www.arc.vt.edu/allocations
PBS -A jianyuan

# Access group. Do not change this line.
#PBS -W group_list=cascades

# Uncomment and add your email address to get an email when your job starts, completes, or aborts
# #PBS -M yourpid@vt.edu
# #PBS -m bea

# Add any modules you might require. This example removes all modules and then adds
# the Intel compiler and mvapich2 MPI modules module. Use the module avail command
# to see a list of available modules.
module purge
module load intel mvapich2

module load Anaconda/2.3.0
module load tensorflow/1.5.0

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR
cd /home/jianyuan/Codes/DQN-DSA

# Below here enter the commands to start your job. A few examples are provided below.
# Some useful variables set by the job:
#  $PBS_O_WORKDIR    Directory from which the job was submitted
#  $PBS_NODEFILE     File containing list of cores available to the job
#  $PBS_GPUFILE      File containing list of GPUs available to the job
#  $PBS_JOBID        Job ID (e.g., 107619.master.cluster)
#  $PBS_NP           Number of cores allocated to the job
# Some useful storage locations (see ARC's Storage documentation for details): 
#  $HOME     Home directory. Use for permanent files.
#  $WORK     Work directory. Use for fast I/O.
#  $TMPFS    File system set up in memory for this job. Use for very fast, small I/O
#  $TMPDIR   Local disk (hard drive) space set up for this job

# Say "Hello world!"
echo "Hello world!" 

# Run the program a.out
# ./a.out
python multiNodeLearning.py


# Run the MPI program mpiProg. The -np flag tells MPI how many processes to use. $PBS_NP
# is an environment variable that holds the number of processes you requested. So if you
# selected nodes=2:ppn=32 above, $PBS_NP will hold 64.
#mpirun -np $PBS_NP ./mpiProg

# Run the OpenMP program ompProg with 32 threads (there are 32 cores per Cascades Haswell node)
export OMP_NUM_THREADS=32
./ompProg

exit;

