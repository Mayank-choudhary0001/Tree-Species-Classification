
# 🌳 Tree Species Classification 

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*An advanced AI-powered web application for tree species identification, location-based recommendations, and intelligent forestry insights using machine learning and computer vision.*

[🚀 Quick Start](#-quick-start) • [📖 Features](#-features--capabilities) • [🧠 ML Architecture](#-machine-learning-architecture) • [📋 Setup Guide](#-complete-setup--usage-guide) • [🤝 Contributing](#-contributing)

</div>

## 📑 Table of Contents

- [🎯 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [✨ Features & Capabilities](#-features--capabilities)
- [�️ Dataset & Data Sources](#️-dataset--data-sources)
- [📋 Complete Setup & Usage Guide](#-complete-setup--usage-guide)
- [🚀 Deployment Options](#-deployment-options)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📧 Contact & Support](#-contact--support)

---

## 🎯 Overview

The **Tree Species Classification** is a comprehensive machine learning solution that combines:
- **🌍 Location Intelligence**: K-NN based tree species recommendations
- **🔍 Species Discovery**: Geographic distribution analysis
- **📸 Image Classification**: CNN-powered visual tree identification
- **📊 Data Analytics**: Insights from 1.38M+ tree records

Built with modern ML frameworks and deployed as an interactive web application.

---
## DEMO Screen Shots
<img width="1913" height="983" alt="Screenshot 2025-07-26 022320" src="https://github.com/user-attachments/assets/86c4b359-f1b2-4447-a737-99c323e80d35" />
<img width="1915" height="955" alt="Screenshot 2025-07-26 022359" src="https://github.com/user-attachments/assets/a6d5f9a9-3de8-4a3e-9336-6873b38f4bdf" />
<img width="1918" height="904" alt="Screenshot 2025-07-26 022444" src="https://github.com/user-attachments/assets/3ca17733-f965-44bf-a33d-54782aed7cfc" />
<img width="1903" height="874" alt="Screenshot 2025-07-26 022557" src="https://github.com/user-attachments/assets/1e9844fa-7678-48ec-9f22-224189c381cc" />

---
## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- pip package manager
- 4GB+ RAM (for CNN model loading)

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Mayank-choudhary0001/Tree-Species-Classification.git
cd TREE_SPECIES_CLASSIFICATION

# Install dependencies
pip install -r requirements.txt

# Download the CNN model (255MB)
# Note: The CNN model is not included in the repository due to size limitations
# You can train your own using the tree_CNN.ipynb notebook or contact the author


---

## ✨ Features & Capabilities

### 🌲 1. Smart Location-Based Recommendations
- **Input**: GPS coordinates, tree diameter, native status, city/state
- **Output**: Top 5 most likely tree species for the location
- **Algorithm**: K-Nearest Neighbors with geospatial clustering
- **Use Case**: Urban planning, forestry management, biodiversity studies

### 📍 2. Species Distribution Mapping  
- **Input**: Select any tree species from dropdown
- **Output**: Geographic distribution and common locations
- **Features**: City-wise prevalence analysis
- **Use Case**: Conservation planning, habitat studies

### 📷 3. AI-Powered Image Classification
- **Input**: Upload tree images (leaves, bark, full tree)
- **Output**: Species prediction with confidence scores
- **Technology**: Custom CNN trained on 30+ species (255MB model)
- **Accuracy**: ~26% on validation set (challenging real-world dataset)
- **Note**: CNN model file not included in repo due to size - train using `tree_CNN.ipynb`

---

## 🗄️ Dataset & Data Sources

### 📊 Tree Metadata Repository
| **Attribute** | **Details** |
|---------------|-------------|
| **Source** | Municipal tree surveys from 50+ U.S. cities |
| **Total Records** | ~1.38 million georeferenced trees |
| **Coverage** | Louisville, Chicago, NYC, LA, and more |
| **Key Fields** | Species names, GPS coordinates, diameter, native status |
| **Time Period** | 2018-2022 survey data |

**Key Data Columns:**
- `common_name`: Tree species (e.g., Bur Oak)
- `scientific_name`: Botanical name (e.g., Quercus macrocarpa)  
- `latitude_coordinate`, `longitude_coordinate`: GPS location
- `city`, `state`, `address`: Geographic identifiers
- `native`: Whether the tree is native to the area
- `diameter_breast_height_CM`: Tree measurement standard

### 🖼️ Image Classification Dataset
| **Attribute** | **Details** |
|---------------|-------------|
| **Species Count** | 30 common species |
| **Total Images** | 1,454 samples |
| **Resolution** | Standardized to 224×224 pixels |
| **Augmentation** | Rotation, zoom, flip transformations |
| **Quality** | Real-world conditions (varying lighting, angles) |

**Dataset Structure:** Folder-based organization with each folder named after tree species for supervised learning.

---


**Technical Details:**
- **Algorithm**: scikit-learn `NearestNeighbors`
- **Distance Metric**: Euclidean distance in scaled feature space
- **Features**: Geographic + environmental + biological attributes
- **Performance**: Sub-second response time for 1.38M records

### 🧠 CNN Image Classifier
```
Input: 224×224×3 RGB Image
    ↓
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
    ↓
Conv2D(128) → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → Dropout(0.5) → Dense(30)
    ↓
Output: Species Probability Distribution
```

**Model Specifications:**
- **Framework**: TensorFlow/Keras
- **Architecture**: Sequential CNN with dropout regularization
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam (learning_rate=0.001)
- **Training**: 50 epochs with validation monitoring
- **Model Size**: 255MB (`basic_cnn_tree_species.h5`)


## 🛠️ Technical Implementation

### 📁 Project Structure
```
TREE_SPECIES_CLASSIFICATION/
├── 📊 Data Processing
│   ├── 5M_trees.ipynb          # Train recommender system
│   └── tree_CNN.ipynb          # Train CNN classifier
├── 🚀 Production Application  
│   |
│   └── requirements.txt        # Dependencies
├── 🤖 Trained Models
│   ├── tree_data.pkl          # Processed dataset (1.9MB)
│   ├── scaler.joblib          # Feature scaler (<1MB)
│   ├── nn_model.joblib        # KNN model (1MB)
│   └── basic_cnn_tree_species.h5  # CNN model (255MB)
└── 📚 Documentation
    |__ README.md              # This file
    
```



## 📋 Complete Setup & Usage Guide

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/Mayank-choudhary0001/Tree-Species-Classification.git
cd TREE_SPECIES_CLASSIFICATION

# Create virtual environment (recommended)
python -m venv tree_env
tree_env\Scripts\activate  # Windows
# source tree_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Model Training (Optional - Models Included)
```bash
# Train recommender system (generates tree_data.pkl, scaler.joblib, nn_model.joblib)
jupyter notebook 5M_trees.ipynb

# Train CNN classifier (generates basic_cnn_tree_species.h5)
jupyter notebook tree_CNN.ipynb
```


### Known Limitations
- **CNN Accuracy**: Limited by small training dataset (1,454 images for 30 classes)
- **Image Quality**: Performance varies with lighting, angle, and image clarity
- **Species Coverage**: Limited to 30 common North American species

### Future Improvements
- [ ] Expand image dataset with data augmentation techniques
- [ ] Include international tree species and locations
- [ ] Implement ensemble methods for improved accuracy
- [ ] Add leaf shape and bark texture analysis
- [ ] Mobile application development

---

## 🚀 Deployment Options

### Local Development
```bash
streamlit run streamlit_integrated.py
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas for Contribution
- 🖼️ **Dataset Expansion**: Add more tree species images
- 🌍 **Geographic Coverage**: Include international tree data
- 🧠 **Model Improvements**: Enhance CNN architecture
- 🎨 **UI/UX**: Improve web interface design
- 📱 **Mobile Support**: Responsive design enhancements

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 Python style guidelines
- Add docstrings for new functions
- Include unit tests for new features
- Update documentation for API changes

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Support

**Author**: Mayank Choudhary  
**GitHub**: https://github.com/Mayank-choudhary0001  
**Repository**: [TREE_SPECIES_CLASSIFICATION](https://github.com/Mayank-choudhary0001/Tree-Species-Classification.git)


## 🙏 Acknowledgments

- **Data Sources**: Municipal tree survey departments
- **ML Frameworks**: TensorFlow, scikit-learn communities  
- **Web Framework**: Streamlit development team
- **Image Dataset**: Contributing photographers and botanical databases

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for urban forestry and environmental conservation

</div>
