# DHL Supply Chain Capstone Project – Spare Parts Identification Using Computer Vision

## 🚚 Project Overview
This capstone project was conducted in collaboration with **DHL Supply Chain**, with the goal of improving the defective spare parts receiving process in warehouses. The focus was to automate part identification using image recognition and OCR to reduce manual effort, speed up processing, and improve accuracy in spare part validation.

---

## 🎯 Problem Statement
Warehouse operators often face delays and errors when identifying defective spare parts due to:
- Lack of labeling on returned parts
- Image quality inconsistencies (blur, lighting, angle)
- Manual lookup and matching against master part lists

The objective was to develop a machine learning model capable of accurately identifying parts from photographs taken during the receiving process.

---

## 🔧 Solution Approach
The solution combines **image recognition**, **OCR**, and **data augmentation** to classify and extract part information:

### Key Steps:
1. **Data Collection**  
   - Gathered image dataset from warehouse operations (varied angles, lighting conditions)
   - Preprocessed and labeled training data

2. **Data Augmentation**  
   - Used `Augmentor` to create diverse training images (rotation, blur, zoom)

3. **Model Development**  
   - Implemented a **YOLOv8**-based object detection model to identify parts
   - Integrated with **Google Vision API** for OCR to extract serial or part numbers

4. **Evaluation & Improvement**  
   - Achieved **90% detection accuracy** after tuning hyperparameters and optimizing feature extraction
   - Reduced misclassification rate by **30%** through model iteration

5. **Deployment Plan**  
   - Designed framework for integration with warehouse system for future scalability

---

## ✅ Project Outcomes
- 📸 **90% accuracy** in identifying parts from images using YOLOv8
- 🕒 **40% reduction** in spare part processing time at receiving stations
- 🔁 Enhanced repeatability through automated workflows and reduced human dependency
- 📦 Improved traceability of unlabelled parts, contributing to smoother inventory reconciliation

---

## 📁 Repository Structure
- `notebooks/`: Model training, tuning, and evaluation scripts (YOLOv8 + OCR)
- `data/`: Sample augmented images and raw dataset structure
- `reports/`: Performance metrics, misclassification logs, and final presentation slides
- `scripts/`: Utility functions for OCR, image pre-processing, and Google API integration

---

## 🧰 Tools & Libraries
- Python, OpenCV, Augmentor  
- YOLOv8 (Ultralytics)  
- Google Cloud Vision API  
- Pandas, NumPy, Matplotlib  
- Jupyter Notebooks, Google Colab  

---

## 🤝 Team & Stakeholders
- Project conducted as part of ASU MSBA Capstone program  
- Partnered with DHL Supply Chain’s engineering and operations team  
- Guided by faculty mentors from W. P. Carey School of Business  

---

## 📌 Conclusion
This project demonstrates how advanced computer vision techniques and OCR can significantly streamline logistics processes. It showcases the potential of AI-powered solutions in operational optimization and supply chain automation.
