# ML-DL-Algorithms-From-Scratch 🧠

A comprehensive collection of Machine Learning and Deep Learning algorithms implemented from scratch using Python. Each algorithm is implemented in its own branch for clarity and educational purposes.

## 🌟 Overview

This repository contains pure Python implementations of popular ML/DL algorithms, using only NumPy for numerical computations. The goal is to provide clear, well-documented implementations that help understand the mathematical principles behind each algorithm.

## 🌳 Branch Structure

Each algorithm has its own branch with the following naming convention:
```
algo/{algorithm-name}
```

### Current Implementations:

#### 🤖 Machine Learning
- `algo/linear-regression` - Linear Regression with gradient descent
- `algo/logistic-regression` - Logistic Regression for binary classification
- `algo/decision-trees` - Decision Trees for classification and regression
- `algo/random-forest` - Random Forest ensemble method
- `algo/kmeans` - K-means clustering
- `algo/knn` - K-Nearest Neighbors
- `algo/svm` - Support Vector Machines
- `algo/naive-bayes` - Naive Bayes Classifier

#### 🧠 Deep Learning
- `algo/neural-network` - Basic Neural Network with backpropagation
- `algo/cnn` - Convolutional Neural Network
- `algo/rnn` - Recurrent Neural Network
- `algo/lstm` - Long Short-Term Memory networks
- `algo/autoencoder` - Basic Autoencoder

## 📦 Requirements

```
numpy==1.21.0
matplotlib==3.4.2
scikit-learn==0.24.2  # For datasets and evaluation metrics only
jupyter==1.0.0        # For running example notebooks
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ML-DL-Algorithms-From-Scratch.git
cd ML-DL-Algorithms-From-Scratch
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
git checkout algo/algorithm-name
```

## 📘 Structure of Each Algorithm Branch

Each algorithm branch contains:
```
.
├── algorithm/
│   ├── __init__.py
│   ├── model.py          # Core algorithm implementation
│   └── utils.py          # Helper functions
├── examples/
│   ├── basic_usage.ipynb # Basic usage examples
│   └── advanced.ipynb    # Advanced applications
├── tests/
│   └── test_model.py     # Unit tests
├── README.md             # Algorithm-specific documentation
└── requirements.txt
```

## 🧪 Testing

Each algorithm includes its own test suite. To run tests:

```bash
python -m pytest tests/
```

## 📊 Example Usage

Basic example of using the Linear Regression implementation:

```python
from algorithm.model import LinearRegression

# Initialize the model
model = LinearRegression(learning_rate=0.01, iterations=1000)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch for your algorithm:
```bash
git checkout -b algo/your-algorithm-name
```
3. Implement your algorithm following the repository structure
4. Add tests and documentation
5. Submit a Pull Request

## 📝 Algorithm Documentation

Each algorithm branch contains its own detailed README with:
- Mathematical explanation
- Implementation details
- Usage examples
- Performance characteristics
- References to relevant papers

## 📖 Resources

- [Mathematics for Machine Learning](https://mml-book.github.io/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- Original papers for each algorithm are linked in their respective branches

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
⭐️ Star this repository if you find it helpful!