# Face Recognition System using PCA + ANN

A machine learning-based face recognition system built using classical dimensionality reduction techniques and a neural network classifier.  
The project demonstrates an end-to-end ML pipeline including preprocessing, feature extraction, model training, optimization, and deployment using Streamlit.

---

## 🚀 Project Overview

This system recognizes faces from a trained dataset using:

- **Principal Component Analysis (PCA)** for feature extraction
- **Linear Discriminant Analysis (LDA)** for class separation
- **Artificial Neural Network (MLPClassifier)** for classification

The application is deployed using **Streamlit**, allowing users to upload an image and receive real-time predictions.

---

## 🧠 Machine Learning Pipeline

1. Convert image to grayscale  
2. Resize image to **100 × 100**  
3. Flatten image into feature vector  
4. Apply PCA (50 components)  
5. Apply LDA transformation  
6. Predict using trained ANN model  
7. Display predicted class with confidence score  

---

## 📊 Model Optimization

Originally trained on 300×300 images, which produced large model sizes (>100MB).

Optimized by:
- Reducing image size to **100×100**
- Reducing PCA components to **50**
- Final model size: **~3.9MB**

This makes the model lightweight and suitable for deployment.

---

## 🛠 Tech Stack

- Python
- NumPy
- OpenCV
- Scikit-learn
- Streamlit

---

## 📁 Project Structure

PCA_ANN_Face_Recognition/
│
├── app.py
├── pca.pkl
├── lda.pkl
├── ann_model.pkl
├── class_names.pkl
├── requirements.txt
├── PCA_ANN_Face_recognition.ipynb
└── README.md


---

## 💻 Run Locally

1. Clone the repository
2. Install dependencies:

pip install -r requirements.txt


3. Run the Streamlit app:

streamlit run app.py


4. Upload a face image to test prediction.

---

## 🌍 Deployment

The application is designed for deployment using:

- Streamlit Cloud
- Render
- Any Python-supported cloud service

---

## ⚠️ Notes

- The model performs **closed-set classification**.
- If confidence is below threshold (65%), the system labels the face as **Unknown Person**.
- Best performance occurs with images similar to the training dataset.

---

## 📌 Key Learnings

- Dimensionality reduction using PCA
- Feature space optimization
- Classical ML pipeline design
- Model size optimization for deployment
- Real-world Git and version control handling

---

## 👨‍💻 Author

Saikumar Pasunuti  
Aspiring GenAI & Machine Learning Engineer
