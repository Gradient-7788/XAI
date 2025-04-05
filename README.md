# Towards a Rigorous Evaluation of XAI Methods on Time Series

This project explores **Explainable AI (XAI)** techniques for 1D time series classification using deep learning models. It applies three popular attribution methods from the [Captum](https://captum.ai) library to analyze and compare the decision-making processes of two models: **SimpleCNN** and **ResNet1D**.

We use **Deletion and Insertion Curves** to assess how accurately each attribution method reflects the model’s reasoning.

---

## 📂 Dataset

The dataset used is a 1D time series classification dataset.  

> **Example**:  
> Dataset: FordA (from [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/))  
> - 3601 training samples, 1320 testing samples  
> - Input shape: `(1, 500)`  
> - Number of classes: 2  
> - Preprocessing: Normalized to range [0, 1]

---

## Models

Two different 1D models are implemented and trained on the dataset:

### 1. `SimpleCNN`
- A lightweight convolutional neural network.
- Architecture:
  - Conv1D → ReLU → MaxPool1D → Dropout → Linear → Softmax

### 2. `ResNet1D`
- A 1D version of the ResNet architecture.
- Uses residual blocks to capture deeper temporal features.

These models are assumed to be trained and available as `model_cnn` and `model_resnet`.

---

## 📌 Explainability Methods

Three attribution methods are used from the **Captum** library:

| Method               | Description                                                 |
|----------------------|-------------------------------------------------------------|
| `IntegratedGradients` | Computes average gradients along a straight-line path from a baseline to the input. |
| `Saliency`            | Computes the gradient of the output with respect to the input features. |
| `DeepLift`            | Compares input activation to a reference and propagates differences. |

Each method helps determine which parts of the input most influenced the model's prediction.

---

## 📉 Deletion and Insertion Curves

To evaluate the *faithfulness* of attributions, the project uses **Deletion and Insertion** metrics:

- **Deletion**: Gradually remove the most important features (by zeroing them) and observe the drop in prediction confidence.
- **Insertion**: Start from a blank input and gradually add the most important features to observe the confidence rise.

Each attribution method is tested on both models using a single test sample.

### Visualization

- A **2×3 grid of plots** is generated:
  - 2 rows: Models (SimpleCNN, ResNet1D)
  - 3 columns: Attribution methods

Each subplot shows:
- X-axis: Fraction of features perturbed
- Y-axis: Prediction probability for the correct class
- Red line: Deletion curve
- Green line: Insertion curve

---

## 📁 File Structure

| File/Function              | Description |
|---------------------------|-------------|
| `models_dict`             | Dictionary with model names and instances |
| `attr_methods`            | Dictionary of Captum attribution method classes |
| `deletion_insertion_curve(...)` | Computes the Deletion or Insertion scores |
| `plot_deletion_insertion_ax(...)` | Plots a single subplot for one model-method pair |
| `plt.subplots(...)`       | Generates all 6 plots for visual comparison |

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install torch torchvision captum matplotlib numpy
```

2. Load the dataset (ECG200) and define your models.

3. Run the attribution and evaluation script.

---

## 📎 File Overview

- `model.py`: Defines `SimpleCNN` and `ResNet1D` architectures
- `attributions.py`: Applies Captum-based attribution methods
- `deletion_insertion.py`: Contains deletion & insertion evaluation logic
- `main.py`: Loads data, computes attributions, and plots results

---

## 📌 References

- UCR Time Series Archive: http://www.timeseriesclassification.com/
- Captum Library: https://captum.ai/
- Paper: *Towards a Rigorous Evaluation of XAI Methods on Time Series*
- Paper: *Axiomatic Attribution for Deep Networks* (Integrated Gradients)
- Paper: *Bayesian XAI methods towards a robustness-centric approach to Deep Learning: an ABIDE I study*

---

## 🧪 Future Work

- Add support for SHAP & LIME
- Use multi-class datasets from UCR
- Implement faithfulness metrics like AUC or area between insertion & deletion curves

