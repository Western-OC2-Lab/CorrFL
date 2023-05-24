# CorrFL: Correlation-based Neural Network Architecture for Unavailability Concerns in a Heterogeneous IoT Environment. 

This code provides the implementation of a Correlation Neural Network in Federated Learning to address model heterogeneity and unavailability concern in the Federated Learning environment. All the code documentation and variable definition is in accordance with the content of the manuscript published in 
**IEEE Transactions on Network and Service Management**: 
I. Shaer and A. Shami, "CorrFL: Correlation-based Neural Network Architecture for Unavailability Concerns in a Heterogeneous IoT Environment, " *IEEE Transactions on Network and Service Management*, doi: 10.1109/TNSM.2023.3278937.

Before experimenting with the code, the following steps need to be implemented: 

1. Download the data employed for this work, which can be found using this link: https://zenodo.org/record/3774723#.ZGEhAHbMKUl. Since we only ran out experiments on room00, retain the data of this room, which can be found by following `data_cleaning/dirty_room/room00` directory. 
2. Create the `datasets` folder in the root directory (on the same level as `src`) and its sub-folders `dirty_data`
3. Relocate the remaining files in the directory `datasets/dirty_data/room00_preprocessed`, which is defined in the `dataset_dir` **variable** in `src/generate_granular_data.py`. Make sure to rename the `csv` files to their corresponding node names. For example, `room00_THP-CO2_924_20190101-20191231.csv` to `node_924`.
4. Create `datasets/dirty_data/room00_gran5_fine` folder directory as declared in `generate_granular_data.py`.
5. Run the `generate_granular_data.py` using the following command: `python generate_granular_data.py`. This results in creating datasets in the `datasets/dirty_data/room00_gran5_fine` folder, which will be used throughout the experiments.
6. Create the `results` folder in the root directory and its sub-folders `figures`, `insights`, `models`, and `stats`
7. Run `driver.py` using the following command `python driver.py`.


This paper highlights the problem of **Oblique Federated Learning**, which is halfway the **Vertical Federated Learning** and **Horizontal Federated Learning**, featuring non-uniformity in the feature space of local agents
while these local agents share some and not all of the feature space. Therefore, this paper and code expose a novel problem in the space of **Federated Learning** and propose exciting research questions for practitioners and 
researchers to address. 

# Requirements
The requirements are included in the `requirements.txt` file. To install the packages included in this file, use the following command: `pip install -r requirements.txt`

# Contact-Info

Please feel free to contact me for any questions or research opportunities. 
- Email: shaeribrahim@gmail.com
- Gihub: https://github.com/ibrahimshaer and https://github.com/Western-OC2-Lab
- LinkedIn: [Ibrahim Shaer](https://www.linkedin.com/in/ibrahim-shaer-714781124/)
- Google Scholar: [Ibrahim Shaer](https://scholar.google.com/citations?user=78fAJ_IAAAAJ&hl=en) and [OC2 Lab](https://scholar.google.com/citations?user=ICvnj9EAAAAJ&hl=en)
