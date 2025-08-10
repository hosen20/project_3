
 ## Wiser 2025 Quantum Project
 ## Project 3 - Quantum Algorithm as a PDE Solver for Computational Fluid Dynamics (CFD)



### Team Information:
Team Member 1:
 - Full Name: Maham Khalid 
 - Womanium Program Enrollment ID (see Welcome Email, format- WQ24-xxxxxxxxxxxxxxx): WQ24-GWc9PP8Qd8pJLht

Team Member 2:
 - Full Name: Abdullah Kazi
 - Womanium Program Enrollment ID (see Welcome Email, format- WQ24-xxxxxxxxxxxxxxx): WQ24-oHZVwR3GSw8vubn


Team Member 3:
 - Full Name: Rishikesh Gokhale
 - Womanium Program Enrollment ID (see Welcome Email, format- WQ24-xxxxxxxxxxxxxxx): WQ24-SGc1RqUvLvpgqL4


Team Member 4:
 - Full Name: Sara Latreche 
 - Womanium Program Enrollment ID (see Welcome Email, format- WQ24-xxxxxxxxxxxxxxx): WQ24-BdB77D9iOFILIIS


### Project Solution:
**Womanium Quantum AI Project 2024**
# :space_invader: Quantum-AI-for-Climate - Team Q-MARS 

  - [Project Description](#Projectdescription)
- [ Setup Instructions](#SetupInstructions)
-  [Team Introduction](#team-introduction)
- [Acknowledgement](#Acknowledgement)





## Project Description
As Dr. Youzuo Lin succinctly put it: ** “Seismic waves are currently the only effective tool that can penetrate the entire Earth, and seismic inversion (tomography) is used to obtain the structural information of the Earth” ** [1]. This structural information has led to the discovery of oil wells and the monitoring of geothermal energy extracted from the Earth's crust, along with other subsurface features. By accurately mapping these structures, seismic inversion enables more effective resource exploration and management, significantly impacting industries like energy and environmental science.

Seismic inversion is a straightforward yet powerful technique. It involves transforming routine reflectivity data, which typically highlights interfaces like geological boundaries, into rock properties known as impedance—calculated by multiplying sonic velocity and bulk density. In conventional seismic sections, high-amplitude reflections mark geological formation boundaries, such as the top of a reservoir, making them well-suited for structural analysis. In contrast, inverted data provides insights into internal rock properties, such as lithology, porosity, and fluid types (like brine or hydrocarbons). This makes seismic inversion particularly valuable for stratigraphic interpretation and reservoir characterization.

Seismic impedance reveals extensive geological information about the subsurface. Accurate seismic impedance inversion is essential for interpreting seismic data effectively [2]. This process can be divided into two categories: linear impedance inversion and nonlinear impedance inversion. Linear impedance inversion assumes a direct, proportional relationship between seismic data and impedance. However, nonlinear impedance inversion accounts for more complex, non-proportional relationships by employing nonlinear models and algorithms, capturing intricate geological variations.

The two major challenges of seismic inversion are that it is often an ill-posed problem with multiple possible solutions that can fit the data, making it difficult to obtain a unique and accurate result [3]. Additionally, noise in the seismic data can obscure crucial information and degrade the quality of the inversion, further complicating the process.

This project explores the application of quantum computing and artificial intelligence (AI) to seismic data analysis, aiming to improve geophysical modeling, resource exploration, and environmental monitoring. Seismic inversion, a critical technique in geophysics, traditionally relies on converting seismic reflectivity data into rock properties such as impedance, which provides detailed information about subsurface structures. However, the process is complex, often ill-posed, and susceptible to noise, leading to multiple potential solutions.

To address these challenges, this project leverages cutting-edge quantum methods, including Variational Quantum Classifiers (VQC), Quantum Autoencoders (QAE), Quantum Circuit-based Clustering, and Quantum Phase Estimation (QPE). Each method focuses on different aspects of seismic data processing, such as pattern recognition, feature extraction, data compression, clustering, and phase estimation. By integrating quantum circuits with classical models, the project aims to enhance the accuracy, efficiency, and scalability of seismic data analysis.

The key goals include normalizing and preprocessing seismic and impedance trace data, designing quantum circuits for data encoding, applying quantum-enhanced algorithms like Kernel Principal Component Analysis (Kernel PCA) for clustering, and evaluating the overall effectiveness of these quantum methods. The project also involves visualizing seismic traces, impedance traces, and statistical wavelets to gain deeper insights into the data.

Through the implementation and testing of these quantum circuits, the project seeks to demonstrate the potential of quantum computing to revolutionize seismic data analysis, offering more precise and efficient solutions for geophysical challenges.

### Methods and objectives
#### 1. **Variational Quantum Classifier (VQC) Neural Network**
**Method Description:**
This method involves using a hybrid classical-quantum approach to enhance seismic data processing through a Variational Quantum Classifier (VQC). The goal is to predict acoustic impedance from normalized seismic traces by integrating quantum circuits with classical neural network models. The quantum circuit utilizes angle embedding and entangling layers to capture complex data relationships, which are then processed by classical layers. A corresponding fully classical neural network is also explored in order to compare results and effectiveness of quantum method.

**Objective:**
- Predict acoustic impedance from seismic traces using a VQC integrated with classical neural networks.
- Evaluate the potential benefits of this hybrid approach in improving seismic data analysis.

#### 2. **Quantum Autoencoders (QAE)**
**Method Description:**
This method applies Quantum Autoencoders (QAE) for seismic impedance inversion. The goal is to leverage quantum circuits to compress and reconstruct seismic trace data, which can then be used to predict impedance values. The QAE model is trained on seismic trace data and evaluated based on its performance in data reconstruction, focusing on minimizing errors. Corresponding classical method is included as a benchmark.

**Objective:**
- Implement a QAE model to compress and reconstruct seismic trace data for predicting impedance values.
- Assess the effectiveness of quantum circuits in seismic data compression and reconstruction.

#### 3. **Quantum Circuit for Seismic Data Clustering**
**Method Description:**
This method explores the use of quantum circuits for enhancing seismic data clustering through Kernel Principal Component Analysis (Kernel PCA). The process includes normalizing and preprocessing seismic and impedance trace data, followed by the application of quantum circuits for data encoding and clustering. The effectiveness of these quantum-enhanced circuits is evaluated, with visualizations generated to provide insights into the data.

**Objective:**
- Develop quantum circuits to encode seismic data and apply Kernel PCA for effective clustering.
- Evaluate the impact of quantum-enhanced circuits on seismic data analysis and visualization.

#### 4. **Quantum Phase Estimation (QPE)**
**Method Description:**
This method integrates Quantum Phase Estimation (QPE) with classical seismic data analysis to enhance the interpretation of seismic traces and impedance data. The objectives include designing a QPE circuit to estimate phase information from seismic data, optimizing parameters, and comparing quantum outcomes with classical results. The method aims to demonstrate the feasibility of quantum techniques in improving seismic analysis.

**Objective:**
- Implement and optimize a QPE circuit for phase estimation in seismic data.
- Compare the effectiveness of QPE against classical methods in seismic analysis and reservoir characterization.
## Results
#### Quantum Methods:
![autoencddd](https://github.com/user-attachments/assets/0fdadf7e-e06b-43c7-befc-a7e5ba6e5b04)
![autoc](https://github.com/user-attachments/assets/8c250c6a-ce8d-4f9b-afd5-7e5e1c98fc65)
![PCA](https://github.com/user-attachments/assets/b0b34a8e-5448-439d-9aad-af1949d33c74)
![pcaaaaaa](https://github.com/user-attachments/assets/d9633ce4-d58a-464d-a3d8-99cb6134d9da)
![PCCCCAAAAAA](https://github.com/user-attachments/assets/613faf40-118c-469c-9947-f7cdc47fdeb4)
![PCAAAGG](https://github.com/user-attachments/assets/66097c96-7ab1-4512-a919-02f1b47e12ba)
![qcbnnnn](https://github.com/user-attachments/assets/93e00011-bb49-4e62-a371-7061a224887b)
![bnnhjs](https://github.com/user-attachments/assets/bf8d7d7a-e7cb-44d4-b1db-97df4275a8dc)
![autoencoder](https://github.com/user-attachments/assets/ff3f74b5-9857-4468-bb8f-e89e517dd58f)
![kernelPCA](https://github.com/user-attachments/assets/8c50b15d-0727-4cb3-8e9b-21b3e60368a8)
![qccccnn](https://github.com/user-attachments/assets/5261be4a-5091-4bd3-bb21-dd743994110b)
![bcnnnnnnn](https://github.com/user-attachments/assets/64a42f87-c9dc-47dd-84f3-941cab51580d)
![QPEEEEE](https://github.com/user-attachments/assets/a18b21e9-d4f0-40d8-a4ab-4fdcbcb96b58)
![QEPPPPPPEPPE](https://github.com/user-attachments/assets/85d26d49-533f-4b76-8be9-08fff868a82c)
![QEPPPSPPPPS](https://github.com/user-attachments/assets/9ed2f62e-5e34-419a-bc0c-cbcd4d8bacf8)

#### Classical Methods:
![1](https://github.com/user-attachments/assets/6879a6c1-f85a-41cc-8fe0-af1ead56a9ba)
![2](https://github.com/user-attachments/assets/fb376610-f68b-489f-b7bf-e756545674af)
![3](https://github.com/user-attachments/assets/46b1e48c-8d5e-4132-96b0-09178320c756)
![4](https://github.com/user-attachments/assets/ec155e3e-0164-4057-b359-d4b371ec02d3)







## Setup Instructions

#### 1. **Clone the Repository**

   Open your terminal or command prompt and run the following command to clone the repository:

   ```bash
   git clone 
```
Navigate into the project directory:
```bash
cd 
```
#### 2. **Create a Python Virtual Environment**
Create a virtual environment to manage dependencies:

```bash
python -m venv venv
```
#### 3. **Activate the Virtual Environment**
On Windows:
```bash
venv\Scripts\activate
```
On macOS and Linux:
```bash
source venv/bin/activate
```
#### 4. **Install Dependencies**
Install the required dependencies listed in the ['requirements.txt'] file:

#### 5. **Download the Dataset**
Download the 
Or 
Directly use a processed data file from the directory.
- ## Usage of the Project

### Uploaded Files


- Images - Contains image files related to project.
- `ModelClassical` - Directory with classical model files.
- `ModelsQML` - Directory with quantum machine learning model files.
- `README.md` - This file, providing an overview and instructions for the project.
- `Requirements.txt` - Contains the list of dependencies required for running the project.
- `docs` - Directory containing additional documentation and related files.

### Notebooks

- `ModelsQML` - Notebook for training using Quantum algorithms.
- `ModelClassical` - Notebook for training using classical algorithms.

### Data Files

- `Data` - Contains processed data for training.



## Team Introduction
**Team Name:** Q-M.A.R.S

|   **Member Names**| **Abdullah Kazi**                      | **Maham Khalid** | **Rishikesh Gokhale** |   **Sara latreche** |
|----------------|-----------------------------------|----------------------------|----------------------------|----------------------------|
| **GitHub ID**  | AbdullahKazi500                   | maham-khalid1      | rishihg      |   Latrechesara|



----------------------

## Acknowledgement
We would like to extend our heartfelt gratitude to WOMANIUM
Thank you for the opportunity to participate in this event, which has not only enriched our knowledge and skills but also connected us with a network of like-minded professionals and enthusiasts.


### Project Presentation Deck:
Presentation Video (Without Audio) linked [here](https://drive.google.com/file/d/1mFgLBOBZYL77u7ospQwYhx0wLa-EQ5hR/view?usp=drive_link)

Slide deck and project descriptions linked [here](https://github.com/maham-khalid1/Quantum-AI-for-Climate/tree/main/docs)

See project presentation guidelines [here](https://docs.google.com/document/d/13nWF8AxFAfFYTWEYPT3BpPdYkqtxxSAjmuXj_zcMh-E/edit?usp=sharing)


