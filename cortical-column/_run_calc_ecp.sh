#!/bin/bash -l
#SBATCH -q debug
#SBATCH --array 1-1
#SBATCH -N 10
#SBATCH -t 00:30:00
#SBATCH -J calc_ecp
#SBATCH -L SCRATCH
#SBATCH -C knl
#SBATCH --mail-user mikelam.us@berkeley.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output "/global/cfs/cdirs/m2043/mikelam/cortical-column/runs/slurm/%A_%a.out"
#SBATCH --error "/global/cfs/cdirs/m2043/mikelam/cortical-column/runs/slurm/%A_%a.err"

set -e

echo
echo "Started at" `date`
echo


cd $CCHOME

SIM_JOBID=40776658

SIM_ARRAY_TASK=1
SIM_JOBNUM=${SIM_JOBID}/${SIM_ARRAY_TASK}
SIM_RUNDIR=$CCDATAROOT/${SIM_JOBNUM}
T_SIM=25000 # This is the # of simulation timesteps
TASKS_PER_NODE=2
CHUNKSIZE=$(( ${T_SIM}/${SLURM_NNODES}/${TASKS_PER_NODE} ))
# GROUPBY=layer_ei # **currently commented out below**

JOBNUM=${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
RUNDIR=$CCDATAROOT/$JOBNUM
OUTDIR=${SIM_RUNDIR}/output
OUTFILE=$OUTDIR/ecp_${SIM_JOBID}_via_${SLURM_ARRAY_JOB_ID}.nwb
IM_FILE=$OUTDIR/im.h5

# We need to create the rundir for this slurm run b/c that's where we
# write temp output before creating the .nwb file
mkdir -p $RUNDIR/output 

echo $OUTFILE

# OpenMP settings
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

cmd="srun --label -N ${SLURM_NNODES} --ntasks-per-node ${TASKS_PER_NODE} python calc_ecp.py \
     --jobnum $JOBNUM \
     --sim-jobnum ${SIM_JOBNUM} \
     --block Simulation_25khz \
     --chunksize ${CHUNKSIZE} \
     --local-chunksize 50 \
     --outfile $OUTFILE \
     --im-file ${IM_FILE}"
echo $cmd
$cmd
     


echo
echo "Finished at" `date`
echo

