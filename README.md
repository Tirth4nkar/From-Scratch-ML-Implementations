# Machine Learning From Scratch 🧠

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="C:\Users\NoobMaster\Documents\Projects\ML Projects\MachineLearningFromScratch\docs\trendy-design-icon-of-machine-learning-vector.jpg"/>
</a>

This repository contains pure Python implementations of popular machine learning, using only NumPy for numerical computations.
The goal is to provide clear, well-documented implementations that help understand the mathematical principles behind each algorithm.

## 🚀 Getting Started
1. Clone the repository:
```bash
git clone https://github.com/yourusername/From-Scratch-ML-Implementations.git
cd CustomNet
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Switch to the algorithm branch you want to explore:
```bash
git checkout -b <your_branch_name>
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch for your algorithm:
```bash
git checkout -b feat/your-feature-name
```
3. Implement your algorithm following the repository structure
4. Add tests and documentation
5. Submit a Pull Request


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         machine_learning_from_scratch and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── machine_learning_from_scratch   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes machine_learning_from_scratch a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------