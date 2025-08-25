# Robust Image Classifier with Neural Collapse-Inspired OOD Detection

This project implements a **robust image classification system in PyTorch** that not only classifies known classes with high accuracy but also detects and rejects **Out-of-Distribution (OOD)** inputs it has never seen before.  

The core of this project is a **ResNet-50 classifier** trained on the **Animals with Attributes 2 (AWA2)** dataset, coupled with a **state-of-the-art, post-hoc OOD detector** based on the principles from the research paper:  
👉 *Detecting Out-of-distribution through the Lens of Neural Collapse*  

📌 A live version of this project is available on Kaggle:  
[**Kaggle Notebook**](https://www.kaggle.com/code/karmveersinghdhounsi/classification-with-ood/edit)

---

## 🚀 Project Overview
In real-world applications, ML models often face inputs outside their training distribution. A standard classifier will still try to assign a label, usually with high confidence—leading to **critical failures**.  

This project builds a system that can confidently say: **“I don’t know.”**

- **Trained on:** 50 animal classes from AWA2  
- **Evaluated against:** Non-animal objects (e.g., airplanes, trucks) from CIFAR-10  

---

## ✨ Key Features
- **Standard Image Classifier:**  
  ResNet-50 trained on AWA2 animal classes.  

- **State-of-the-Art OOD Detection:**  
  Implements the **Neural Collapse Inspired (NCI)** method, analyzing internal network geometry to detect OOD samples.  

- **Fair & Rigorous Evaluation:**  
  Strict train/validation/test splits ensure unbiased performance reporting.  

- **Detailed Performance Metrics:**  
  - OOD Rejection  
  - ID Detection  
  - Conditional Classification Accuracy  

---

## 🧪 Methodology
The project follows a **scientifically rigorous, multi-step workflow**:

1. **Data Preparation**  
   - AWA2 split: 70% train / 10% validation / 20% test  
   - CIFAR-10 split for OOD validation and testing  

2. **Model Training**  
   - ResNet-50 trained **only** on AWA2 animals  
   - No exposure to OOD data during training  

3. **NCI Setup**  
   - Extracts final layer weight vectors (**w_c**)  
   - Computes global mean of training features (**μ_G**)  
   - Uses these to calculate **NCI scores**  

4. **Optimal Threshold Tuning**  
   - Mixed validation set (AWA2 + OOD) used to determine best NCI threshold  

5. **Final Evaluation**  
   - Tested on unseen AWA2 test + OOD test set  

---

## 📊 Final Results
Performance on **unseen test data**:

| Metric                         | Score   | Description |
|--------------------------------|---------|-------------|
| **OOD Rejection Accuracy**     | 95.16%  | Correctly rejected non-animal images |
| **ID Detection Accuracy**      | 92.18%  | Correctly identified in-distribution animals |
| **Conditional Classification** | 92.21%  | Correct classification among accepted animals |

---

## 🛠 Tech Stack
- **Language:** Python  
- **Framework:** PyTorch, Torchvision  
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib, PIL  
- **Datasets:** Animals with Attributes 2 (AWA2), CIFAR-10  

---

## 📂 How to Run
1. **Setup Environment**  
   Install required dependencies:
   ```bash
   pip install torch torchvision numpy pandas scikit-learn matplotlib pillow
