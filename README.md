# D2Cell

------------
D2Cell-pred is a machine learning-based computational toolkit for predicting modification targets in microorganisms.

## Dependencies

We used the following Python packages for core development. We tested on `Python 3.9`.

| name              | version   |
|-------------------|-----------|
| numpy             | 1.24.4    |
| pandas            | 2.0.3     |
| networkx          | 3.1       |
| tqdm              | 4.66.5    |
| torch             | 2.4.0     |
| torch-geometric   | 2.5.3     |
| scipy             | 1.10.1    |
| seaborn           | 0.13.2    |
| scikit-learn      | 1.3.2     |


### Usage

Clone codes and download necessary data files
- (1). Download the D2Cell-pred package
```shell
git clone https://github.com/LiLabTsinghua/D2Cell.git
```
- (2). Download required Python package
```shell
pip install -r requirements.txt
```
- (3). Download and unzip the [model parameters](https://drive.google.com/file/d/1XPvHyERKNqgMAqjVy3yobzZaYQEuL84_/view?usp=sharing) under D2Cell
- (4). Run Code/D2Cell-pred Model/predict demo.ipynb demo

## Web Server

-------
We also provide an dataset web server: [D2Cell](https://digitallifethu.com/d2cell).

Contact
-------

-   Feiran Li ([@feiranl](https://github.com/feiranl)), Tsinghua University, Shenzhen, China
-   Xiongwen Li ([@xiongwenL](https://github.com/xiongwenL)), Tsinghua University, Shenzhen, China
