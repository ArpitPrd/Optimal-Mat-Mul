## How to Run the Code


To compile and generate all binaries:
```
make
```

This builds both:

```
./baseline/sw_baseline
./optimized/sw_opt
```

To run the performance evaluation on all input sizes and store the results:
```
python3 runner.py
```

All results will be automatically saved in:
```
results.csv
```