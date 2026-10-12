![Plots Screenshot](images/Screenshot%202026-10-12%20125000.png)
# 🐾 WildLens AI — Animal Image Classification

> A CNN-powered image classification project that predicts which of **six animal classes** appears in an image.

## 🎯 Problem Statement
Identifying animal species from images manually can be time-consuming. The goal of this project is to build an image-classification model that automatically predicts an animal category from an uploaded image.

## 💡 Solution
Built and trained a **Convolutional Neural Network (CNN)** using a dataset organized into six classes: **Bear, Elephant, Horse, Lion, Panda, and Tiger**. The trained model is then used in an interactive Streamlit application for image uploads and predictions.

## 🧰 Tech Stack
- **Python** — project implementation
- **TensorFlow / Keras** — image pipeline and CNN model
- **NumPy, Pandas** — data handling and inspection
- **Matplotlib** — image visualization and prediction display
- **Scikit-learn** — classification metrics and confusion matrix
- **Streamlit** — interactive web interface
- **Pillow (PIL)** — image handling in the app

## 🔬 Workflow & Techniques
1. **Dataset inspection** — checked class folders, image formats, dimensions, and invalid files.
2. **Preprocessing** — decoded images as RGB, resized to **160 × 160**, and used aspect-ratio-preserving padding.
3. **Dataset splitting** — separated data into training, validation, and test sets.
4. **Data pipeline** — used `tf.data`, batching, parallel mapping, and prefetching.
5. **Data augmentation** — random horizontal flips, rotations, and zooms to improve training variety.
6. **CNN architecture** — convolution layers with ReLU, max pooling, dense layers, dropout, and a six-class softmax output.
7. **Training** — used Adam, sparse categorical cross-entropy, validation monitoring, and early stopping.
8. **Evaluation** — assessed performance on held-out test data using classification metrics and a confusion matrix.
9. **Prediction & deployment** — predicted individual images and built a Streamlit interface to show class scores.

## 📊 Dataset
The dataset contains six animal categories:

| Class | Class | Class |
|---|---|---|
| 🐻 Bear | 🐘 Elephant | 🐎 Horse |
| 🦁 Lion | 🐼 Panda | 🐅 Tiger |

The dataset had approximately **1,500 images** during inspection. Exact counts can vary if files are added or removed.

## ✨ Features
- 🖼️ Upload an image and preview it
- 🧠 Predict one of six animal classes
- 📈 View confidence/probability scores for all classes
- 📥 Download a prediction summary
- ⚡ Cache the model so it doesn't reload on every interaction
- 🎨 Responsive, dark-themed Streamlit UI

## 📈 Results
The model achieved approximately **82% accuracy** during experimentation. The final result should be reported using the held-out test-set accuracy from the completed evaluation, since training or validation accuracy can differ from test accuracy.

## 🚀 Run the App
1. Clone or download this project.
2. Put `animal_classification_model.keras` in the same folder as `app.py`.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start Streamlit:

   ```bash
   python -m streamlit run app.py
   ```

## 📁 Project Structure
```text
animal-cnn-classifier/
├── notebook.ipynb                  # Dataset exploration, training and evaluation
├── main_data/                      # Six class folders of animal images
├── animal_classification_model.keras
├── app.py                          # Streamlit application
├── requirements.txt
└── README.md
```

## ⚠️ Notes & Limitations
- The model recognizes only the six classes used during training.
- Predictions may be unreliable for blurry, heavily cropped, unusual, or out-of-scope images.
- A high confidence score is not a guarantee of correctness.
- Keep the class-name order in the app consistent with the label order used during training.
- The app preprocessing must match the notebook preprocessing, including the model's internal `Rescaling(1./255)` layer.

## 🔮 Future Improvements
- 🌿 Add more animal classes and more diverse images
- 🧪 Improve generalization with more data and transfer learning
- 📱 Deploy the app online
- 🧾 Add prediction history and batch image prediction

---

### ⭐ If you found this project interesting, give the repository a star!
