# Artificial intelligence in healthcare: past, present and future

- [Paper](paper.pdf)

## Notes

- Human physicians will not be replaced by machines in the foreseeable future, but AI can definitely assist physicians to make better clinical decisions or even replace human judgement in certain functional areas of healthcare (eg, radiology)
- Two major categories in AI:
  - The first category includes machine learning (ML) techniques that analyse structured data such as imaging, genetic and EP data. In the medical applications, the ML procedures attempt to cluster patients’ traits, or infer the probability of the disease outcomes.
  - The second category includes natural language processing (NLP) methods that extract information from unstructured data such as clinical notes/medical journals to supplement and enrich structured medical data.
- Inputs to ML algorithms include patient ‘traits’ and sometimes medical outcomes of interest.
  - A patient’s traits commonly include baseline data, such as age, gender, disease history and so on, and disease-specific data, such as diagnostic imaging, gene expressions, EP test, physical examination results, clinical symptoms, medication.
  - Besides the traits, patients’ medical outcomes are often collected: disease indicators, patient’s survival times and quantitative disease levels, for example, tumour sizes.
- An NLP pipeline comprises two main components: (1) text processing and (2) classification. Through text processing, the NLP identifies a series of disease-relevant keywords in the clinical notes based on the historical databases. Then a subset of the keywords are selected through examining their effects on the classification of the normal and abnormal cases. The validated keywords then enter and enrich the structured data to support clinical decision making.
- Major obstacles:
  - The first hurdle comes from the regulations. Current regulations lack of standards to assess the safety and efficacy of AI systems.
  - The second hurdle is data exchange. In order to work well, AI systems need to be trained (continuously) by data from clinical studies. However, once an AI system gets deployed after initial training with historical data, continuation of the data supply becomes a crucial issue for further development and improvement of the system. Current healthcare environment does not provide incen- tives for sharing data on the system.
