# Fruit Quality and Size Classification
This project is a part of the **Algoritmos y Programacion III** course in the Applied Artificial Intelligence Master, Universidad Icesi, Cali Colombia. 

#### -- Project Status: Active

## Contributing Members

**Team Leader: [Sara Lucia Diaz Puerta](https://github.com/SaraLuciaa)(@SaraLuciaa)**
**Instructor: Uram Anibal Sosa**

#### Other Members:

|Name     |  Email   | 
|---------|-----------------|
|[Jose Daniel Guzman Castrillon](https://github.com/DanielGuzman13)| danielguz1305@gmail.com        |
|[Faiber Stiven Piedrahita Perlaza](https://github.com/Fazb3r) |     faiberpiedrahita@gmail.com    |

## Contact
* Feel free to contact the team leader or the instructor with any questions or if you are interested in contributing!


## Project Intro/Objective
The purpose of this project is to develop an automatic quality classification and size estimation system for fruits and vegetables using computer vision. Manual classification of fresh products based on size, ripeness, or visible defects is slow, subjective, and prone to errors. This system addresses this issue by analyzing images to assign a quality category and estimate the relative size of the product.

### Methods Used
* Computer Vision
* Machine Learning / Deep Learning
* Image Segmentation
* Data Visualization

### Technologies
* Python
* Streamlit
* OpenCV
* NumPy, Pandas
* Scikit-Learn
* Plotly, Matplotlib

## Project Description
The system automatically classifies quality categories and estimates physical dimensions of fruits. The model was trained using a dataset that combines public sources and custom collections, and it is evaluated using rigorous classification and regression metrics. We employ Streamlit for an interactive dashboard where users can upload fruit images, calibrate physical size using pixels-to-cm ratios, view the segmented contours of the fruit, and review historical analytics of analyzed batches.

## Getting Started
Instructions for contributors
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](sources) within this repo.
3. Data processing/transformation scripts are being kept [here](src/data)
4. Follow setup [instructions](docs/instalacion.md)

## Featured Notebooks/Analysis/Deliverables
* [Jupyter Notebook for Exploration](notebooks/experiment_1.ipynb)
* [System Architecture Overview](docs/arquitectura.md)
* [Installation Guide](docs/instalacion.md)
