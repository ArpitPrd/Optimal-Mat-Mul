## How to run the code


### For the full run test

```
chmod +x run_tests.sh run_all_tests.sh
./run_all_tests.sh

# generates all results in gemm_results.csv
```

### For a single run

```
chmod +x run.sh
./run.sh
```

### For plotting

```
pip install numpy matplotlib pandas
python3 plot_results.py
# generates plots in plots/*
```