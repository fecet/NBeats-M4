# Reproduce performance of NBeats in M4 Dataset

Link to [[paper](https://arxiv.org/abs/1905.10437)]. 

## Steps to reproduce

- Download M4 Dataset at [Mcompetitions/M4-methods](https://github.com/Mcompetitions/M4-methods) 
- Placed it at somewhere and modifiled global variable `DATA_PATH` in `data.py` properly, the directory structure is:
    ```
    Dataset
    ├── M4-info.csv
    ├── Test
    │   ├── Daily-test.csv
    │   ├── Hourly-test.csv
    │   ├── Monthly-test.csv
    │   ├── Quarterly-test.csv
    │   ├── Weekly-test.csv
    │   └── Yearly-test.csv
    └── Train
        ├── Daily-train.csv
        ├── Hourly-train.csv
        ├── Monthly-train.csv
        ├── Quarterly-train.csv
        ├── Weekly-train.csv
        └── Yearly-train.csv
    ```
- run the .ipynb file.


## Results

There is only results for Yearly dataset and interpretable model at the moment. Check

## Credit:

- [philipperemy/n-beats](https://github.com/philipperemy/n-beats)
- [ElementAI/N-BEATS](https://github.com/ElementAI/N-BEATS)
- [Mcompetitions/M4-methods](https://github.com/Mcompetitions/M4-methods)
