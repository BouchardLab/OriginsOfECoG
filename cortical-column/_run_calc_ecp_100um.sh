#!/bin/bash -l
#SBATCH -q premium
#SBATCH --array 10-11
#SBATCH -N 50
#SBATCH -t 2:00:00
#SBATCH -J calc_ecp_100um
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH --mail-user vbaratham@berkeley.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output "/global/cscratch1/sd/vbaratha/cortical-column/runs/slurm/%A_%a.out"
#SBATCH --error "/global/cscratch1/sd/vbaratha/cortical-column/runs/slurm/%A_%a.err"
# #DW persistentdw name=calc_ecp_100um
# mkdir $DW_JOB_STRIPED/output

set -e

echo
echo "Started at" `date`
echo


cd $SCRATCH/cortical-column

SIM_JOBID=32529070

SIM_ARRAY_TASK=1
SIM_JOBNUM=${SIM_JOBID}/${SIM_ARRAY_TASK}
SIM_RUNDIR=runs/${SIM_JOBNUM}
T_SIM=60000 # This is the simulation length in ms
TASKS_PER_NODE=2
CHUNKSIZE=$(( ${T_SIM}/${SLURM_NNODES}/${TASKS_PER_NODE}*10 ))
echo "CHUNKSIZE" $CHUNKSIZE


JOBNUM=${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
RUNDIR=runs/$JOBNUM
OUTDIR=${SIM_RUNDIR}/output
OUTFILE=$OUTDIR/ecp_200um_${SIM_JOBID}_via_${SLURM_ARRAY_JOB_ID}.nwb
# TMP_OUTPUT_DIR=$DW_JOB_STRIPED/output
TMP_OUTPUT_DIR=$OUTDIR # We are just taking the individual slice h5 files as the final output
IM_FILE=$DW_JOB_STRIPED/im.h5
mkdir -p $RUNDIR/output # needed b/c we write temp files here

echo $OUTFILE

# OpenMP settings
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun --label -N ${SLURM_NNODES} --ntasks-per-node ${TASKS_PER_NODE} python calc_ecp_100um.py \
     --jobnum $JOBNUM \
     --sim-jobnum ${SIM_JOBNUM} \
     --block Simulation_v1 \
     --chunksize ${CHUNKSIZE} \
     --local-chunksize 50 \
     --outfile $OUTFILE \
     --electrodes-file runs/32223414/1/network/electrodes.csv \
     --array-task ${SLURM_ARRAY_TASK_ID} \
     --tmp-output-dir ${TMP_OUTPUT_DIR} \
     # --im-file ${IM_FILE}

# Run accumulate_ecp_groups.py separately when all finish

echo
echo "Finished at" `date`
echo

