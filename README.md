# DiagnosAI
Building AI course project : AI Health Navigator for Rare Genetic Disease Detection and Treatment Guidance

## Summary

Millions of individuals suffer from chronic, unexplained, or misdiagnosed conditions, especially rare, autoimmune, and genetic diseases. Traditional diagnostic pathways often fail due to overlapping symptoms, fragmented data, and a lack of awareness, resulting in prolonged suffering and ineffective treatments. DiagnosAI is an AI-powered platform designed to help individuals actively participate in their diagnostic journey by analyzing symptoms, medical histories, genetic data, and drug experiences.

The platform’s goal is to enhance early detection, provide personalized diagnostic recommendations, guide patients toward appropriate specialists, and assist in identifying effective treatments. The system encourages patients to engage in their health discovery process, empowering them to make informed decisions about their treatment options and medical care.

## Background

### Problem
  Chronic, unexplained, and misdiagnosed conditions often lead to years of frustration and poor health outcomes. Rare diseases, autoimmune disorders, and genetic conditions are difficult to diagnose because their symptoms often overlap with more common illnesses. This results in a delayed diagnosis, incorrect treatments, and unnecessary suffering. DiagnosAI aims to bridge this gap by providing early detection, symptom tracking, and expert-recommended treatment pathways.
  
### Personal Motivation
The inspiration for this project came from my own personal experience with misdiagnoses. After being misdiagnosed with various common conditions such as irritable bowel syndrome, migraines, and others since childhood, I was finally diagnosed at the age of 39 with a rare form of Familial Mediterranean Fever (FMF) through Next Generation Sequencing (NGS), after enduring years of suffering. Due to the late diagnosis, my body had already been irreparably damaged, and the high inflammation caused by the condition led to autoimmune diseases, psoriasis, brain issues, and many other complications. This made the process of treating and managing the diseases extremely complicated and nearly impossible. After conducting my own research, I realized that the treatments prescribed for autoimmune diseases were actually exacerbating my FMF, and my treatment had effectively reached a dead end.

This experience inspired me to create a system that helps others, who are suffering from unexplained pain, recognize their condition early, receive the correct diagnosis, and get on the right path to treatment.

### Importance
This project aims to tackle the diagnostic delay and inaccuracies often associated with rare diseases and misdiagnosed conditions. Many of these patients have been ignored by the healthcare system or have been wrongly treated for years. DiagnosAI provides a public-facing platform that allows individuals to input their symptoms and medical histories, receive personalized diagnostic insights, and be guided toward the appropriate specialists, tests, and treatments. The platform will also provide insights into potential drug conflicts, helping patients avoid harmful interactions.

## Data sources and AI methods

### Data Sources:
- **MIMIC-III / MIMIC-IV**: De-identified patient data for training symptom-diagnosis models.
- **Orphanet**: Clinical and genetic data for building the rare disease knowledge base.
- **ClinVar & Ensembl**: Genetic variant databases to link mutations with diseases.
- **DrugBank & RxNorm**: Information on drugs and interactions to identify treatment conflicts.
- **PubMed / Medline**: Medical literature for enriching knowledge graphs and NLP models.
- **Patient Advocacy Groups**: Data from registries and case studies for training and validating the models.

### AI Methods:
- **NLP (Natural Language Processing)**: To process symptom descriptions and extract useful features (e.g., using **spaCy**).
- **Machine Learning (ML)**: Algorithms like **Random Forest** and **XGBoost** for diagnosing based on symptoms.
- **Deep Learning**: **LSTM** or **Transformer models** for analyzing symptom progression over time.
- **Knowledge Graphs**: Mapping relationships between symptoms, diseases, and treatments to improve accuracy.
- **Rule-based + ML Hybrid Models**: For predicting drug interactions and conflicts.

### Use of Data
The data will be used to train AI models that detect patterns in symptoms, flag genetic and autoimmune disease indicators, and identify drug interactions. The knowledge graph will continue to evolve as more data is added, improving diagnostic accuracy over time.

## How is it used?

### Target Users:
1. **Patients**: Individuals suffering from chronic, unexplained, misdiagnosed, or rare diseases.
2. **Physicians & Specialists**: Healthcare professionals needing advanced diagnostic support for complex, atypical cases.
3. **Genetic Testing Companies & Research Institutions**: Aiming to enhance patient engagement and optimize data usage for better clinical outcomes. 
   
### Process:
1. **Input Data**: Patients provide symptoms, medical history, and genetic data.
2. **AI Analysis**: The platform processes the data, detecting patterns and generating potential diagnoses.
3. **Recommendations**: DiagnosAI suggests specialist referrals, diagnostic tests, and personalized treatment options.
4. **Drug Conflict Detection**: The system identifies potential drug interactions based on patient history.
5. **Symptom Tracking**: Over time, the system tracks symptom progression and adapts diagnostic recommendations accordingly.

## Challenges

DiagnosAI is not without limitations:
- **Not a replacement for doctors**: It provides insights and suggestions but cannot replace the expertise of medical professionals.
- **Data Quality**: Patient-reported symptoms may be inconsistent, and genetic data may not always be comprehensive.
- **Complexity of Rare Diseases**: Many rare diseases still lack sufficient clinical data and research.
- **Drug Response Variability**: Each individual's response to medications varies, and this needs to be considered in treatment plans.
- **Access Barriers**: Limited access to genetic testing or digital tools in some populations.

### Ethical Considerations:
- Ensuring patient privacy and data security is a top priority when handling sensitive medical information.
- Informed consent must be obtained from all users before collecting or analyzing their data.

## What’s Next?

The future of **DiagnosAI** includes:
- **Wearable Data Integration**: Support for data from wearables (e.g., heart rate, activity) to improve diagnostic recommendations.
- **Clinical Integration**: Partnering with healthcare providers to integrate the tool into clinical workflows.
- **Global Knowledge Sharing**: Creating a platform for sharing knowledge and fostering collaborations in rare disease research.

### Long-term Vision:
- **Drug Discovery**: By analyzing patient data, DiagnosAI aims to contribute to drug discovery, helping pharmaceutical companies create new treatments based on real-world evidence.

## Acknowledgments

This project is inspired by my personal journey in health and the experiences of millions of people who have been misdiagnosed or left undiagnosed. Special thanks to the following organizations and individuals for their contributions to the rare disease community:

* NORD (National Organization for Rare Disorders)
* Global Genes
* Orphanet
* Numerous healthcare professionals and geneticists contributing to research on rare diseases

I would also like to thank **MinnaLearn** and the **University of Helsinki** for helping shape the technical foundations of this project by hosting the Artificial Intelligence Collection courses, and for expanding awareness, knowledge, and the collective ability to apply artificial intelligence in a positive and constructive way.

## Prototype Code

Below is a simple Python prototype of the **DiagnosAI** system, demonstrating how the platform could process patient symptoms and suggest a potential diagnosis using basic machine learning techniques.

### Code Example:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Example symptom-diagnosis data
data = {
    'fever': [1, 0, 1, 1, 0],
    'rash': [1, 0, 1, 1, 0],
    'headache': [0, 1, 1, 0, 1],
    'joint_pain': [1, 1, 0, 0, 1],
    'diagnosis': ['FMF', 'Migraine', 'FMF', 'Psoriasis', 'Migraine']
}

df = pd.DataFrame(data)

# Feature variables and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using RandomForest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Example of predicting for new patient data
new_patient = np.array([[1, 1, 0, 1]])  # Example: Fever, Rash, No headache, Joint pain
predicted_diagnosis = model.predict(new_patient)
print(f'Predicted Diagnosis: {predicted_diagnosis[0]}')
