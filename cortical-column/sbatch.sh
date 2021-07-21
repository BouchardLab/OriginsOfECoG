#!/bin/bash -l
#SBATCH -q debug
#SBATCH --array 1-1
#SBATCH -N 32
#SBATCH -t 0:30:00
#SBATCH -J cortical-column
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --mail-user mikelam.us@berkeley.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output "/global/cfs/cdirs/m2043/mikelam/cortical-column/runs/slurm/%A_%a.out"
#SBATCH --error "/global/cfs/cdirs/m2043/mikelam/cortical-column/runs/slurm/%A_%a.err"

######################
## SETTINGS AND SHIT
######################

# for knl
module unload craype-hugepages2M 


set -e

echo
echo "Started at" `date`
echo

# source /project/projectdirs/m2043/vbaratha/nrn-parallel/parallelnrn.env
cd $CCHOME


RUNDIR=$CCDATAROOT/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
mkdir -p $RUNDIR
# stripe_large $RUNDIR
lfs setstripe $RUNDIR --stripe-count 150

# OpenMP settings 
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


TSTOP=100


####################################
## CHOOSE A NETWORK CREATION METHOD
####################################


## COPY BBP
# srun --label -n 1 python BBP_build_network.py \
#      --network=$RUNDIR/network \
#      --reduce .1 \
#      --ctx-ctx-weight .6e-4 \
#      --ctx-ctx-weight-std 1e-5 \
#      --thal-ctx-weight 2.0e-4 \
#      --thal-ctx-weight-std 2e-5 \
#      --bkg-weight 2.0e-4 \
#      --num-bkg 5000 \
#      --num-thal 5000 \
#      --circuit-file individuals/pathways_P14-14/pathways_mc0_Column.h5
# echo "Finished copying BBP network at" `date`

## REUSE PREVIOUS RUN
RUN=40776658/1
cp -r $CCDATAROOT/$RUN/network $RUNDIR/network
rm -f $RUNDIR/network/*_spikes.csv # use new spike trains
mkdir $RUNDIR/output
echo "reusing network from $RUN"


#########################
##  RUN THE SIMULATION
#########################


srun -n 1 python configure.py \
       --base-config=base_config_ecp.json \
       --network-dir=$RUNDIR/network \
       --output-dir=$RUNDIR/output \
       --config-outfile=$RUNDIR/config.json \
       --stim wn_simulation_ampl_v1_05 \
       --timestep 0.1 \
       --tstop $TSTOP \
       --cells all \
       --nsteps-block 500 \
       --optocell '5e' 


srun --label -n $((${SLURM_NNODES}*64)) -c 1 python run.py $RUNDIR/config.json

#srun python count_layer_segments.py --jobnum ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID} --outfile $RUNDIR/output/layer_slice_counts.json

echo
echo "Finished at" `date`
echo

