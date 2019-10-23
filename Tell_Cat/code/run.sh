time=$(date "+%b%d_%H_%M")
nohup python -u main.py > $time.txt 2>&1 &
