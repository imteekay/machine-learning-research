# Scenario 4: The Hospital Readmission Dilemma

**The Situation:** A hospital administrator is facing penalties from the government because too many patients suffering from heart failure are being readmitted to the hospital within 30 days of their initial discharge. They want to intervene before the patient leaves the first time.

* *Your Task:* Frame this as an ML problem. Keep in mind that false negatives (missing a patient who *will* be readmitted) are much more dangerous here than false positives. How does this impact your metric choices?

---

- Goal: Reduce the number of readmitted patients
- Business metric: readmittion rate (for the hospital admin) - how many patients are coming back after their first discharge within 30 days?
  - Downsides: it takes 30 days to have a complete and reliable measurement for this metric
- Problem: 
  - Problem 1: readmittion means that patients maybe have heart issues but it was not detected on the first analaysis
    - Detection is failing: doctors could not detect the issue from patient exams
    - Exams: Electrocardiogram, Echocardiogram, Blood Tests
  - Problem 2: patients were already diagnosed with heart failure issues and we need to predict if they will be readmitted
- Data:
  - Electrocardiogram (Time-Series): max heart rate (number)
  - Echocardiogram (Video): sequences of images (CNN)
  - Blood Tests (Tabular): colesterol (number), blood sugar (number)
- ML frame: what if we could use an ml model as a diagnostic tool for the medical staff make better decisions
  - Electrocardiogram: detects Arrhythmias (Irregular heartbeats), Prior or Current Heart Attacks, Ischemia (Poor blood flow), Electrolyte imbalances (High or low potassium and calcium levels)
    - Time-series: RNN
    - Multi-class classification or multiple binary classification models
    - Cross entropy loss
    - Precision, recall, F1 score, but more weight on recall as false negative are more dangerous
  - Echocardiogram: detects Heart Valve Disease, Heart Failure, Cardiomyopathy
    - Video: video transformers, 3D CNN
    - Multi-class classification or multiple binary classification models
    - Cross entropy loss
    - Precision, recall, F1 score, but more weight on recall as false negative are more dangerous
  - Blood Tests: detects Acute Heart Damage, Plaque Build-Up Risk, Heart Strain, Systemic Inflammation
    - Tabular: tree-based algorithms
    - Multi-class classification or multiple binary classification models
    - Cross entropy loss
    - Precision, recall, F1 score, but more weight on recall as false negative are more dangerous
  - Data exploration (explore the present): patient clusters/disease gravity clusters
- Target 1: model can trained to detect heart issues
- Target 2: Use the detected heart issues, blood test results, and clinical history to predict if they will be readmitted
