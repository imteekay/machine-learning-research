# Sybil: A Validated Deep Learning Model to Predict Future Lung Cancer Risk From a Single Low-Dose Chest Computed Tomography

[Paper](paper.pdf)

- Trained on LDCTs from the National Lung Screening Trial (NLST): training, dev, and test
- Tested on Massachusetts General Hospital (MGH) and Chang Gung Memorial Hospital (CGMH)
  - For testing, Sybil requires only one LDCT and does not require clinical data or radiologist annotations. 
  - It's interesting how the model learned throught the annotations from training and could figure it out on test sets without those annotations.
- Sybil, that uses a single low-dose chest computed tomography (CT) scan to predict lung cancers occurring 1-6 years after a screen.
- Context
  - Lung cancer screening (LCS) rates have focused on identifying those at the highest risk for lung cancer and directing available resources to screen them
  - Lung cancer diagnoses among never- and lighter-smokers are rapidly rising
- Data
  - Considered any given LDCT positive in terms of future cancer risk if biopsy-confirmed lung cancer was diagnosed within 6 years, independent of presence absence of nodules or other abnormalities on that examination
  - Radiologists jointly annotated suspicious lesions on NLST LDCTs using MD.AI software for all participants who developed cancer within 1 year after an LDCT
- Model
  - Sybil was designed to predict future lung cancer risk using a 3D convolutional neural network architecture
  - Sybil’s outcome is a set of six scores representing calibrated probabilities of lung cancer diagnosis extending 1 to 6 years following the LDCT
- Results
  - CGMH cohorts was similar to its power in the NLST test set
  - It's interesting how it can have similar accuracy using data from people with different ethnicity/race (CGMH)
  - Understanding when the risk score likely relies on the presence of a nodule and when it does not
    - Sybil’s performance was hampered by removing visible nodules
    - To distinguish between cancer detection and future cancer risk, visible lung nodules that were known to be cancerous were removed from the analysis set. It was found that Sybil’s performance was lower on this set but still possessed predictive power.
  - Association between Sybil’s ability to correctly lateralize the location of future cancers and the likelihood that an LDCT receives a high-risk score
  - Sybil may also infer biologically relevant information from LDCT images
    - Traditional clinical risk factors such as smoking duration can be predicted directly from the LDCT images
  - Clinical application: Sybil predicting a high risk score (60% risk percentile) when the Lung-RADS clinical assessment was low risk (scores 1 or 2)
- Limitations
  - gain con dence that it is generalizable: Scans from the NLST, which were obtained in 2002-2004 from US patients who were overwhelmingly White (92%)

## Benefits

- Sybil can predict future lung cancer risk from a single LDCT scan with a high degree of accuracy (Area Under the Curve or AUC) in the range of 0.86 to 0.94 for predicting cancer within one year and 0.74 to 0.81 for prediction within six years in various validation datasets. This suggests it could potentially outperform traditional risk assessment models that rely on clinical factors alone.
- Provide a personalized risk score based on the LDCT image, Sybil could help clinicians determine the optimal screening intervals for individual patients. Those at lower risk might safely extend their screening intervals, while those at higher risk could be monitored more closely. This could lead to more efficient and cost-effective screening programs.
- Sybil's ability to predict risk from the LDCT image itself, without relying on extensive clinical data or radiologist annotations, could potentially expand lung cancer screening to individuals who may not meet current high-risk criteria (e.g., never-smokers who are increasingly being diagnosed with lung cancer).
- Reduced False Positives: This could lead to fewer unnecessary follow-up procedures (like biopsies) and reduce patient anxiety.
- It can learn complex patterns and subtle features in the LDCT images that might not be visible to the human eye
- Sybil is designed to analyze LDCT scans as they become available without requiring additional input from radiologists or manual annotations
