infile=$1
block=Simulation_v0

if [ $# -eq 0 ]
  then
      echo "pass input ECP nwb file as argument to this script"
      exit
fi

python scripts/preprocess.py $infile --block $block --device ECoG --acquisition Raw --first-resample 3200 --final-resample 400 --no-notch --no-car
python scripts/preprocess.py $infile --block $block --device Poly --acquisition Raw --first-resample 3200 --final-resample 400 --no-notch --no-car

for i in `seq 1 6`;
do
    python scripts/preprocess.py $infile --block $block --device ECoG --acquisition L$i --first-resample 3200 \
           --final-resample 400 --no-notch --no-car --dset-postfix L$i
    python scripts/preprocess.py $infile --block $block --device Poly --acquisition L$i --first-resample 3200 \
           --final-resample 400 --no-notch --no-car --dset-postfix L$i
done
