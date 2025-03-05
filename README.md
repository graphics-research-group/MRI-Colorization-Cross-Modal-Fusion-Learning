# Paper Title

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Repo Status](https://img.shields.io/badge/status-active-success)

## 📄 Introduction
This repository contains the code, data, and supplementary materials for the paper:

**"[Structurally Consistent MRI Colorization using Cross-modal Fusion Learning]"**  
_Mayuri Mathur, Anav Chaudhary, Saurabh Kumar Gupta, Ojaswa Sharma_  
[IEEE, ISBI]  

📌 **Abstract:**  
_Medical image colorization can greatly enhance the interpretability of the underlying imaging modality and provide insights into human anatomy. The objective of medical image colorization is to transfer a diverse spectrum of colors distributed across human anatomy from Cryosection data to source MRI data while retaining the structures of the MRI. To achieve this, we propose a novel architecture for structurally consistent color transfer to the source MRI data. Our architecture fuses segmentation semantics of Cryosection images for stable contextual colorization of various organs in MRI images. For colorization, we neither require precise registration between MRI and Cryosection images, nor segmentation of MRI images. Additionally, our architecture incorporates a feature compression-and-activation mechanism to capture organ-level global information and suppress noise, enabling the distinction of organ-specific data in MRI scans for more accurate and realistic organ-specific colorization. Our experiments demonstrate that our architecture surpasses the existing methods and yields better quantitative and qualitative results._

## 📊 Results
### Figures
Below are some key figures from the paper:

![Figure 1](./results/multiresoutputs.png)
*Figure 1: Description of Figure 1.*

![Figure 2](./results/Comparisons.png)
*Figure 2: Description of Figure 2.*

### Tables
Below are some key tables from the paper:

| Metric  | Value |
|---------|-------|
| SSIM | |
| MSSSIM | |
| FSIM  | |
| STSIM  |  |

## 📁 Repository Structure
<!-- ```
📂 project-root
├── 📜 paper.pdf                  # Final version of the paper
├── 📂 code                        # Source code for experiments/simulations
│   ├── script1.py
│   ├── script2.py
│   └── ...
├── 📂 data                        # Sample datasets or preprocessing scripts
├── 📂 results                     # Experimental results, figures, plots
├── 📂 models                      # Pretrained models (if applicable)
├── 📜 requirements.txt            # List of dependencies
├── 📜 README.md                   # This file
└── 📜 LICENSE                     # License information
``` -->

## 🚀 Getting Started
### Prerequisites
Ensure you have the following installed:
- Python >= 3.x
- Required packages listed in `requirements.txt`

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

### Running the Code
To reproduce the experiments:
```bash
python script1.py --arg1 value --arg2 value
```

## 📊 Results
Include key figures, tables, or summaries from the paper. Optionally, provide links to additional results.

## 📑 Citation
If you use this work, please cite:
```bibtex
@article{your_citation,
  author    = {Author Name(s)},
  title     = {Paper Title},
  journal   = {Journal/Conference},
  year      = {Year},
  doi       = {DOI}
}
```

## 📬 Contact
For questions, feel free to reach out to:
- **Your Name** - [Email](mailto:mayurim@iiitd.ac.in)
- Open an issue in this repository.


