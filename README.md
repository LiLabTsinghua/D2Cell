# D2Cell

Introduction
------------

We developed D2Cell-pred, a hybrid model that combines mechanistic and deep learning approaches to predict outcomes for new cell factories. D2Cell-pred takes as input the target product, the GEM structure, and a set of gene modifications, and outputs the predicted impact of these modifications on the product.

Dependencies
------------
We used the following Python packages for core development. We tested on `Python 3.9`.

| name            | version |
|-----------------|---------|
| numpy           | 1.24.4  |
| pandas          | 2.0.3   |
| networkx        | 3.1     |
| tqdm            | 4.66.5  |
| torch           | 2.4.0   |
| torch-geometric | 2.5.3   |
| scipy           | 1.10.1  |
| seaborn         | 0.13.2  |
| scikit-learn    | 1.3.2   |
| ipywidgets      |         |


Usage
------------
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
- (4). Run **Code/D2Cell-pred Model/predict demo.ipynb** demo. This demo applies the D2Cell-pred model to E. coli for target prediction. Users can input Gene IDs and Target Product IDs to evaluate the validity of specific metabolic targets.
> The input fields require specific **ID formats** corresponding to the iML1515 metabolic model:
> - **Gene IDs:** Use the gene ID (e.g., `b0002`) found in **`Data/D2Cell-pred Data/Ecoli/iML1515_Genes.tsv`**.
> - **Product IDs:** Use the specific metabolite ID (e.g., `ala__D_c`) found in **`Data/D2Cell-pred Data/Ecoli/ecoli_product_idx.csv`**.

Web Server
-------
We also provide an dataset web server: [D2Cell](https://digitallifethu.com/d2cell).
A static snapshot of the D2Cell database has now been deposited on Zenodo (https://zenodo.org/records/18240770), ensuring permanent accessibility.

Contact
-------

-   Feiran Li ([@feiranl](https://github.com/feiranl)), Tsinghua University, Shenzhen, China
-   Xiongwen Li ([@xiongwenL](https://github.com/xiongwenL)), Tsinghua University, Shenzhen, China
