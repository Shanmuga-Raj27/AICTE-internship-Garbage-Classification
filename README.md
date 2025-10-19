# AICTE-internship-Shanmugaraj
Garbage Classification Project using AI/ML

# Garbage Classification Using Transfer Learning

## Project Overview
This project implements a deep learning solution for garbage classification using transfer learning with EfficientNetV2B2. The model classifies trash images into 6 different categories to assist in automated waste sorting and recycling processes.

## Dataset Information
- **Dataset Name**: TrashType_Image_Dataset
- **Total Images**: 2,527
- **Number of Classes**: 6
- **Classes**:
  1. cardboard
  2. glass
  3. metal
  4. paper
  5. plastic
  6. trash

### Dataset Split
- **Training Samples**: 2,022 images (80%)
- **Validation Samples**: 505 images (20%)
- **Test Samples**: Split from validation set
- **Batch Size**: 32
- **Image Size**: 224×224 pixels

## Model Architecture
### Base Model
- **Transfer Learning Model**: EfficientNetV2B2
- **Input Shape**: (224, 224, 3)
- **Pre-trained Weights**: ImageNet

### Custom Layers
- Rescaling layer for input normalization
- Global Average Pooling 2D
- Custom classification head for 6-class output

## Key Features
1. **Data Preprocessing**:
   - Automatic dataset loading with TensorFlow
   - Image resizing to 224×224
   - Data augmentation (implied)
   - Train/validation/test split

2. **Class Imbalance Handling**:
   - Uses `compute_class_weight` from scikit-learn
   - Addresses potential class distribution issues

3. **Model Evaluation**:
   - Confusion matrix analysis
   - Classification report
   - Performance metrics

4. **Deployment Ready**:
   - Gradio interface for web deployment
   - Interactive model testing

## Technical Implementation
### Libraries Used
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV (implied)
- **Data Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: scikit-learn metrics
- **Deployment**: Gradio

### Training Configuration
- **Optimizer**: Custom configuration
- **Callbacks**: Implemented for training monitoring
- **Class Weights**: Computed for imbalanced data

## Project Structure
```
Garbage_Classification/
├── Garbage Classification Shanmugaraj (Main).ipynb
├── garbage/TrashType_Image_Dataset/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├
├
└── README.md
```

## Usage
### Training the Model
1. Load the dataset from the specified directory
2. Preprocess images and split into train/validation/test sets
3. Initialize EfficientNetV2B2 with pre-trained weights
4. Add custom classification layers
5. Train with class weights and callbacks
6. Evaluate model performance

### Deployment
The project includes a Gradio interface for:
- Uploading trash images
- Real-time classification
- Probability distribution display
- Interactive web interface

## Performance Metrics
The model evaluation includes:
- Accuracy metrics
- Confusion matrix visualization
- Precision, recall, and F1-score per class
- Classification report

## Applications
- Automated waste sorting systems
- Recycling facility automation
- Environmental education tools
- Smart city waste management

## Developer
**Shanmugaraj**

## Requirements
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Gradio

## Future Improvements
- Expand dataset with more diverse trash items
- Implement real-time classification
- Mobile app deployment
- Integration with IoT waste sorting systems
- Multi-modal classification (combining image and sensor data)

This project demonstrates the practical application of transfer learning for environmental sustainability, providing an efficient solution for automated garbage classification that can significantly improve recycling processes and waste management systems.
