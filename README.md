# Machine learning foundations

Recap projects for the
[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
(DeepLearning.AI and Stanford), which I'm going back through as a refresher. After each course,
I apply its main ideas to a real public dataset, outside finance, with scikit-learn, and write
up the decisions and takeaways.

These are lighter than my [main portfolio projects](https://marlondocouto.com): one notebook
each, honest baselines, a test set used once, and tests for the parts that could go wrong.
They use my own data and code, not the course's labs.

| Project | Course | Concepts | Data |
|---|---|---|---|
| [01 · Airbnb in Rio de Janeiro](01-rio-airbnb/) | 1 · Supervised learning | Linear regression, cost function, gradient descent, learning rate, feature scaling, polynomial features, overfitting, regularization, logistic regression, decision boundary, threshold | Inside Airbnb, OpenStreetMap |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r 01-rio-airbnb/requirements.txt
```
