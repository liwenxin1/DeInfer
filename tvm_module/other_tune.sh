
for batch_size in 8 16 32
do
    python other_tune.py --network  transformer --batch_size $batch_size
done

for batch_size in 50 100
do
    python other_tune.py --network  LSTM --batch_size $batch_size
done