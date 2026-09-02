```
./run_mot_benchmark.sh
./run_mot_benchmark.sh --smoke
./run_mot_benchmark.sh --tracker kf
./run_mot_benchmark.sh --tracker norfair
./run_mot_benchmark.sh --pipeline sequential
./run_mot_benchmark.sh --pipeline pipelined
./run_mot_benchmark.sh --tracker kf --pipeline pipelined
./run_mot_benchmark.sh --method kf
./run_mot_benchmark.sh --method norfair
./run_mot_benchmark.sh --method kf-sequential
./run_mot_benchmark.sh --method kf-pipelined
./run_mot_benchmark.sh --method norfair-sequential
./run_mot_benchmark.sh --method norfair-pipelined
./run_mot_benchmark.sh --smoke --method kf-pipelined
./run_mot_benchmark.sh -- --max-segments 5 --conf 0.5
```
