# Public Histopathology Datasets for Training

This document provides a comprehensive guide to publicly available histopathological image datasets that can be used for training the autoencoder models.

## Major Public Datasets

### 1. **The Cancer Genome Atlas (TCGA)**
- **Source**: [TCGA Portal](https://portal.gdc.cancer.gov/)
- **Content**: 30,000+ whole slide images across 33 cancer types
- **Format**: SVS, TIFF (high resolution)
- **License**: Public domain
- **Quality**: Excellent (clinical grade)
- **Download**: Free with registration

```bash
# Example download using TCGA API
pip install gdctools
gdctools download --cases TCGA-BRCA --data-type "Slide Image"
```

### 2. **PathAI Datasets**
- **Source**: [PathAI GitHub](https://github.com/PathAI/pathml)
- **Content**: Various curated histology datasets
- **Format**: Multiple formats (PNG, TIFF, SVS)
- **License**: Various (check individual datasets)
- **Quality**: High (research grade)

### 3. **Kaggle Histopathology Competitions**
- **Source**: [Kaggle](https://www.kaggle.com/datasets?search=histopathology)
- **Popular Datasets**:
  - **PatchCamelyon**: 327,680 color images (96x96 pixels)
  - **Breast Cancer Histopathology**: 277,524 patches (50x50 pixels)
  - **Colorectal Cancer**: 5,000 histological images
- **Format**: JPG, PNG
- **License**: Various (mostly CC licenses)
- **Quality**: Good to excellent

### 4. **NIH/NCI Digital Slide Archive**
- **Source**: [Digital Slide Archive](https://digitalslidearchive.github.io/digital_slide_archive/)
- **Content**: Thousands of digitized slides
- **Format**: Multiple formats
- **License**: Various (check individual collections)
- **Quality**: Excellent (research institutions)

### 5. **OpenSlide Test Data**
- **Source**: [OpenSlide](https://openslide.org/demo/)
- **Content**: Sample whole slide images for testing
- **Format**: Various WSI formats (SVS, TIFF, etc.)
- **License**: Public domain
- **Quality**: Good (demonstration quality)

## Specialized Research Datasets

### 6. **CAMELYON Challenge Datasets**
- **CAMELYON16**: Sentinel lymph node detection
- **CAMELYON17**: Metastasis detection and classification
- **Source**: [CAMELYON Challenge](https://camelyon17.grand-challenge.org/)
- **Quality**: Excellent (clinical competition data)

### 7. **TUPAC Challenge**
- **Content**: Tumor proliferation assessment
- **Source**: [TUPAC Challenge](http://tupac.tue-image.nl/)
- **Format**: TIFF
- **Quality**: Excellent (clinical grade)

### 8. **BACH Challenge**
- **Content**: Breast cancer histology classification
- **Source**: [BACH Challenge](https://iciar2018-challenge.grand-challenge.org/)
- **Format**: TIFF
- **Quality**: Excellent

## Academic Institution Datasets

### 9. **Stanford Tissue Microarray Database**
- **Source**: [Stanford TMA](https://tma.im/)
- **Content**: Tissue microarray data
- **Quality**: Research grade

### 10. **University Collections**
- Check university medical schools and pathology departments
- Many offer research datasets with proper permissions
- Contact institutions directly for access

## 🛠 Easy Download Scripts

Here are some helper scripts to get you started:

### Kaggle Dataset Downloader
```bash
# Install Kaggle API
pip install kaggle

# Configure API key (get from kaggle.com/account)
mkdir ~/.kaggle
# Place kaggle.json in ~/.kaggle/

# Download PatchCamelyon dataset
kaggle datasets download -d jejjohnson/patchcamelyon

# Download Breast Cancer dataset  
kaggle datasets download -d paultimothymooney/breast-histopathology-images
```

### TCGA Downloader (Python)
```python
import requests
import json
import os

def download_tcga_images(cancer_type="BRCA", max_files=100):
    """Download TCGA slide images for specified cancer type."""
    
    # TCGA API endpoint
    cases_endpt = "https://api.gdc.cancer.gov/cases"
    files_endpt = "https://api.gdc.cancer.gov/files"
    
    # Filter for cases with slide images
    filters = {
        "op": "and",
        "content": [{
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": [f"TCGA-{cancer_type}"]
            }
        }, {
            "op": "in", 
            "content": {
                "field": "files.data_type",
                "value": ["Slide Image"]
            }
        }]
    }
    
    params = {
        "filters": json.dumps(filters),
        "expand": "files",
        "format": "json",
        "size": str(max_files)
    }
    
    response = requests.get(cases_endpt, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data['data'])} cases with slide images")
        
        # Extract file UUIDs
        file_uuids = []
        for case in data['data']:
            for file_info in case['files']:
                if file_info['data_type'] == 'Slide Image':
                    file_uuids.append(file_info['id'])
        
        print(f"Downloading {len(file_uuids)} slide images...")
        
        # Download files
        for i, uuid in enumerate(file_uuids[:max_files]):
            download_url = f"{files_endpt}/{uuid}"
            response = requests.get(download_url)
            
            if response.status_code == 200:
                filename = f"tcga_{cancer_type}_{i+1:03d}.svs"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {filename}")
            else:
                print(f"Failed to download file {uuid}")
    
    else:
        print(f"API request failed: {response.status_code}")

# Usage
download_tcga_images("BRCA", max_files=50)  # Breast cancer
```

## Dataset Preparation Workflow

### Step 1: Choose Your Dataset
```bash
# For beginners - start with Kaggle datasets (smaller, preprocessed)
kaggle datasets download -d paultimothymooney/breast-histopathology-images

# For advanced users - use TCGA (larger, clinical grade)
python download_tcga.py
```

### Step 2: Extract and Organize
```bash
# Extract downloaded data
unzip breast-histopathology-images.zip -d raw_data/

# Use our preparation script
source slide_env/bin/activate
python -m src.training.prepare_data raw_data/ --output_dir histology_data
```

### Step 3: Quality Check
```bash
# Analyze the prepared dataset
python -m src.training.prepare_data histology_data/ --analyze_only
```

## Recommended Starting Datasets

### For Learning/Testing (Small Scale)
1. **PatchCamelyon** - 327K patches, good for initial experiments
2. **Breast Cancer Histopathology** - Well-organized, good documentation

### For Serious Research (Large Scale)
1. **TCGA** - Clinical grade, massive scale, multiple cancer types
2. **CAMELYON datasets** - Competition quality, well-validated

### For Specific Applications
- **Breast cancer**: TCGA-BRCA, BACH Challenge
- **Colorectal cancer**: Colorectal histology datasets on Kaggle
- **General histology**: OpenSlide test data, PathAI collections

## Legal and Ethical Considerations

### Always Check:
- **License terms** - Some datasets require attribution
- **Usage restrictions** - Research vs commercial use
- **Patient privacy** - Ensure proper de-identification
- **Institutional approval** - Check if your institution requires IRB approval

### Best Practices:
- Cite the original dataset in publications
- Follow data sharing agreements
- Respect patient privacy
- Use data only for stated purposes

## Quick Start Command

Here's a complete script to get you started quickly:

```bash
#!/bin/bash
# Quick dataset setup for histopathology autoencoder

echo "Setting up histopathology training data..."

# Method 1: Kaggle (easiest)
if command -v kaggle &> /dev/null; then
    echo "Downloading Kaggle breast cancer dataset..."
    kaggle datasets download -d paultimothymooney/breast-histopathology-images
    unzip breast-histopathology-images.zip -d raw_histology_data/
    
    # Prepare data
    source slide_env/bin/activate
    python -m src.training.prepare_data raw_histology_data/ --output_dir training_data
    
    echo "Data ready for training!"
    echo "Start training with: python -m src.training.train_real_data training_data/train"
else
    echo "Please install Kaggle CLI: pip install kaggle"
fi
```

## Dataset Size Recommendations

| **Purpose** | **Minimum Images** | **Recommended** | **Suggested Dataset** |
|-------------|-------------------|-----------------|----------------------|
| **Learning** | 100+ | 500+ | PatchCamelyon (subset) |
| **Research** | 1,000+ | 5,000+ | TCGA (single cancer type) |
| **Production** | 5,000+ | 10,000+ | TCGA (multiple types) |

Remember: Quality matters more than quantity. 500 high-quality, diverse images will often outperform 5,000 poor-quality or repetitive images!
