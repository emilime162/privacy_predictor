# Privacy Predictor  

Predicting identity-related labels from images to support **privacy-preserving AI**.  

This project builds on the **Visual Privacy Advisor (VPA) dataset** and applies **deep learning methods** (CNNs, transfer learning, and regularization) to predict sensitive identity-related attributes such as **race** and **hair color**.  

---

## 🚀 Motivation  

The rapid growth of image sharing has raised significant **privacy concerns**. Identifiable information (e.g., race, hair color) can be misused, leading to potential **privacy violations**.  

This project aims to:  
- Explore **privacy-preserving AI** approaches.  
- Predict identity-related labels accurately.  
- Provide insights into model interpretability for safer AI deployment.  

---

## 📊 Dataset  

- **Source:** [Visual Privacy Advisor (VPA) dataset](https://doi.org/10.1109/iccv.2017.398)  
- **Size:** 22,167 images annotated with **68 privacy-related attributes**  
- **Subset used in this project:**  
  - 3,000 training images  
  - 200 validation images  
- **Labels considered:**  
  - **Race**  
  - **Hair Color**  

Each image may have multiple labels (**multi-label classification**).  

---

## 🧠 Methods  

1. **Model:** Convolutional Neural Networks (CNNs)  
2. **Transfer Learning:** Pre-trained **MobileNetV2**  
3. **Regularization:** Dropout (0.5)  
4. **Training Strategy:** Early stopping based on validation loss  

---

## 📈 Results  

On the validation set:  

- **Hamming Accuracy:** 95.36%  
- **Precision:** 95.36%  
- **Recall:** 100%  
- **F1 Score:** 97.63%  

Confusion matrix analysis shows reliable predictions, with a tendency toward false positives but no false negatives.  

---

## 🔮 Future Work  

- Apply **Grad-CAM** for interpretability  
- Expand training to the **full dataset**  
- Explore predicting **activity labels** while preserving privacy  

---

## 📂 Repository Structure  

```plaintext
privacy_predictor/
│── data_util.py       # Data loading and preprocessing  
│── model.py           # CNN model definition and training  
│── train.py           # Training pipeline  
│── utils.py           # Helper functions  
│── requirements.txt   # Python dependencies  
│── README.md          # Project overview (this file)  
