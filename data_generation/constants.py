# DEEP SEEK SETTINGS
BASE_URL = ""
API_KEY = ""

EMBEDDING_ENDPOINT = ""
EMBEDDING_API_KEY = ""

# PROMPTS
CONVERT_COLUMNS_PROMPT = """You are a powerful AI with expertise in medicine. 
You are given a dataset with columns that relate to patients where each patient is a row and each column contains different information pertaining to the patient.
As your first task, you are tasking with converting a list of column names that are possibly abbreviated or not easy to understand into a fully understandable name for medical professionals.
Please provide the output as a Python dictionary. 
The list of column names is: {column_names}"""

FROM_JSON_TO_QUESTION_PROMPT_BIOBANK = """You are a powerful AI with expertise in medicine. 
Your task is to generate a detailed and exhaustive text description for a patient.  
You are given all the patient information in a json-format, which contains the clinical attributes and the results from laboratory tests from real world patients. 
The patients in question are patients with cardiovascular disease.  
The reader of the description is an expert witin this particular medical domain. 
The language used in the description should reflect your domain expertise and your medical reasoning capabilities.
Please provide as many details as possible.
You should ONLY include the patient description!
_____
The json-file containing the information from the patient:\n"""

FROM_JSON_TO_QUESTION_PROMPT_SUPPORT = """You are a powerful AI with expertise in medicine. 
Your task is to generate a detailed and exhaustive text description for a patient.  
You are given all the patient information in a json-format, which contains the clinical attributes and the results from laboratory tests from real world patients. 
The patients are critically ill patients from 5 United States medical centers, accessioned throughout 1989-1991 and 1992-1994. Each hospitalized patient met the inclusion and exclusion criteria for nine disease categories: acute respiratory failure, chronic obstructive pulmonary disease, congestive heart failure, liver disease, coma, colon cancer, lung cancer, multiple organ system failure with malignancy, and multiple organ system failure with sepsis. 
The reader of the description is an expert witin this particular medical domain. 
The language used in the description should reflect your domain expertise and your medical reasoning capabilities.
Please provide as many details as possible.
You should ONLY include the patient description!
_____
The json-file containing the information from the patient:\n"""

FROM_JSON_TO_QUESTION_PROMPT_COVID = """You are a powerful AI with expertise in medicine. 
Your task is to generate a detailed and exhaustive text description for a patient.  
You are given all the patient information in a json-format, which contains the clinical attributes and the results from laboratory tests from real world patients. 
The patients are hospitalized COVID patients.
The language used in the description should reflect your domain expertise and your medical reasoning capabilities.
Please provide as many details as possible.
You should ONLY include the patient description!
_____
The json-file containing the information from the patient:\n"""


FROM_PATIENT_DESCRIPTIONS_TO_TASK_PROMPT = (
    system_prompt_for_tasks
) = """### Role ###
Clinical AI analyzing patient outcomes using contrastive case pairs.

### Input Data ###
1) Target patient (labeled {target_outcome})
2) Nearest neighbor who survived
3) Nearest neighbor who died\n
=== CLOSEST SURVIVOR ===
{survivor_description}\n
=== CLOSEST DEATH ===
{death_description}\n
=== TARGET PATIENT ===
{target_description}\n
### Required Analysis ###
1. Comparison:
   a) Identify 1-3 decisive differences between target and NNs
   b) Focus on features present in ALL THREE cases
   c) Flag any conflicting evidence (e.g., "Target aligns with NN1 in [X] but NN2 in [Y]")

2. Label Evaluation:
   a) Assess if {target_outcome} is correct
   b) Confidence score (1-5):
      5 = All evidence strongly agrees
      4 = Most evidence agrees
      3 = Mixed evidence
      2 = Minimal supporting evidence
      1 = No discernible pattern

3. Counterfactual:
   a) Modify one feature present in NNs
   b) Predict outcome change
   c) Justify using specific NN evidence

### Response Format ###
1. Comparison:
   1) Outcome alignment: <Matches Survivor/Matches Death/Uncertain>
   2) Decisive factors:
      1) <Feature>: Target vs NN1 vs NN2
      2) <Feature>: Target vs NN1 vs NN2

2. Label assessment:
   1) Correctness: <Correct/Incorrect/Edge Case>
   2) Confidence: <1-5 with brief rationale>

3. Counterfactual:
   1) Modification: <Feature + change>
   2) Outcome: <Survived/Died>
   3) Evidence: <"Matches [NN] where [feature]=[value]">

### Quality Rules ###
- If confidence ≤2, state: "Uncertain because [reason]"
- Counterfactuals must reference features actually present in NNs
- Use original feature names from input data
"""


FROM_PATIENT_DESCRIPTIONS_TO_TASK_NO_COUNTERFACTUAL_PROMPT = (
    system_prompt_for_tasks
) = """### Role ###
Clinical AI analyzing patient outcomes using contrastive case pairs.

### Input Data ###
1) Target patient (labeled {target_outcome})
2) Nearest neighbor who survived
3) Nearest neighbor who died\n
=== CLOSEST SURVIVOR ===
{survivor_description}\n
=== CLOSEST DEATH ===
{death_description}\n
=== TARGET PATIENT ===
{target_description}\n
### Required Analysis ###
1. Comparison:
   a) Identify 1-3 decisive differences between target and NNs
   b) Focus on features present in ALL THREE cases
   c) Flag any conflicting evidence (e.g., "Target aligns with NN1 in [X] but NN2 in [Y]")

2. Label Evaluation:
   a) Assess if {target_outcome} is correct
   b) Confidence score (1-5):
      5 = All evidence strongly agrees
      4 = Most evidence agrees
      3 = Mixed evidence
      2 = Minimal supporting evidence
      1 = No discernible pattern

### Response Format ###
1. Comparison:
   1) Outcome alignment: <Matches Survivor/Matches Death/Uncertain>
   2) Decisive factors:
      1) <Feature>: Target vs NN1 vs NN2
      2) <Feature>: Target vs NN1 vs NN2

2. Label assessment:
   1) Correctness: <Correct/Incorrect/Edge Case>
   2) Confidence: <1-5 with brief rationale>

### Quality Rules ###
- If confidence ≤2, state: "Uncertain because [reason]"
- Counterfactuals must reference features actually present in NNs
- Use original feature names from input data
"""

# DICTIONARIES

DICTIONARY_TO_CLINICAL_NAMES_BIOBANK = {
    "Sex": "Sex",
    "Age": "Age (Years)",
    "Weight (kg)": "Weight (Kilograms)",
    "Height (cm)": "Height (Centimeters)",
    "Smoking amount (cigarettes/day)": "Smoking Amount (Cigarettes per Day)",
    "Atrial fibrillation": "Atrial Fibrillation Diagnosis",
    "Chronic kidney disease": "Chronic Kidney Disease Diagnosis",
    "Rheumatoid arthritis": "Rheumatoid Arthritis Diagnosis",
    "Drug status: Anti-diabetic": "Anti-diabetic Medication Use",
    "Drug status: Anti-hypertensives": "Anti-hypertensive Medication Use",
    "History of diabetes": "Diabetes History",
    "Drug status: Lipid-lowering": "Lipid-lowering Medication Use",
    "Drug status: Birth Control Pill": "Oral Contraceptive Use",
    "Glucose (mmol/l)": "Blood Glucose Level (mmol/L)",
    "HbA1c (%)": "Hemoglobin A1c (HbA1c) Percentage",
    "White cell count (x10^9/l)": "White Blood Cell Count (x10^9/L)",
    "Creatinine (\\B5mol/l)": "Serum Creatinine (μmol/L)",
    "Triglycerides (mmol/l)": "Triglyceride Level (mmol/L)",
    "Uric acid (\\B5mol/l)": "Uric Acid Level (μmol/L)",
    "Cystatin-c (mg/l)": "Cystatin C Level (mg/L)",
    "SBP (mmHg)": "Systolic Blood Pressure (mmHg)",
    "Urine Microalbumin (mg/L)": "Urine Microalbumin Concentration (mg/L)",
    "CRP (mg/l)": "C-Reactive Protein (CRP) Level (mg/L)",
    "Familiy history of CVD": "Family History of Cardiovascular Disease (CVD)",
    "Drug status: atypical antipsychotic medication": "Atypical Antipsychotic Medication Use",
    "Drug status: steroid tablets": "Corticosteroid Medication Use",
    "Do you have migraines?": "Migraine History",
    "Severe mental illness?": "Severe Mental Illness Diagnosis",
    "Systemic lupus erythematosus (SLE)": "Systemic Lupus Erythematosus (SLE) Diagnosis",
    "Total cholesterol": "Total Cholesterol Level (mmol/L)",
    "HDL": "High-Density Lipoprotein (HDL) Cholesterol Level (mmol/L)",
    "Ethnicity": "Ethnicity",
    "event": "Clinical Event Occurrence",
    "time_to_event": "Time to Clinical Event (Days)",
}

DICTIONARY_TO_CLINICAL_NAMES_SUPPORT = {
    "Unnamed: 0": "Patient Identifier",
    "sex": "Gender",
    "ARF/MOSF w/Sepsis": "Acute Renal Failure or Multiple Organ System Failure with Sepsis",
    "COPD": "Chronic Obstructive Pulmonary Disease",
    "CHF": "Congestive Heart Failure",
    "Cirrhosis": "Cirrhosis Diagnosis",
    "Coma": "Coma Status",
    "Colon Cancer": "Colon Cancer Diagnosis",
    "Lung Cancer": "Lung Cancer Diagnosis",
    "MOSF w/Malig": "Multiple Organ System Failure with Malignancy",
    "ARF/MOSF": "Acute Renal Failure or Multiple Organ System Failure",
    "Cancer": "Cancer Diagnosis",
    "under $11k": "Annual Income Under $11,000",
    "$11-$25k": "Annual Income $11,000 to $25,000",
    "$25-$50k": "Annual Income $25,000 to $50,000",
    ">$50k": "Annual Income Over $50,000",
    "white": "Ethnicity - White",
    "black": "Ethnicity - Black",
    "asian": "Ethnicity - Asian",
    "hispanic": "Ethnicity - Hispanic",
    "num.co": "Number of Comorbidities",
    "edu": "Education Level (Years)",
    "avtisst": "Average Time in ICU Stay (Days)",  # Assumed based on common medical abbreviations
    "hday": "Hospitalization Day",  # Days since admission
    "diabetes": "Diabetes Diagnosis",
    "dementia": "Dementia Diagnosis",
    "meanbp": "Mean Arterial Blood Pressure (mmHg)",
    "wblc": "White Blood Cell Count (cells/mm³)",
    "hrt": "Heart Rate (beats per minute)",
    "resp": "Respiratory Rate (breaths per minute)",
    "temp": "Temperature (°C)",
    "pafi": "PaO2/FiO2 Ratio (mmHg)",  # Measure of oxygenation
    "alb": "Serum Albumin (g/dL)",
    "bili": "Serum Bilirubin (mg/dL)",
    "crea": "Serum Creatinine (mg/dL)",
    "sod": "Serum Sodium (mEq/L)",
    "ph": "Blood pH Level",
    "glucose": "Blood Glucose (mg/dL)",
    "bun": "Blood Urea Nitrogen (mg/dL)",
    "urine": "24-Hour Urine Output (mL)",
    "adlp": "Activities of Daily Living - Physical (ADLP Score)",
    "adls": "Activities of Daily Living - Self-care (ADLS Score)",
    "death": "Patient Death Indicator (0=Alive, 1=Deceased)",
    "d.time": "Time to Death (Days post-admission)",
    "age": "Age (Years)",
}

DICTIONARY_TO_CLINICAL_NAMES_COVID = {
    "is_dead": "Deceased Status",
    "Days_hospital_to_outcome": "Days from Hospital Admission to Outcome",
    "Age": "Age (Years)",
    "Sex": "Sex",
    "Race": "Race",
    "SG_UF_NOT": "State of Notification (Brazilian UF Code)",
    "Fever": "Fever",
    "Cough": "Cough",
    "Sore_throat": "Sore Throat",
    "Shortness_of_breath": "Shortness of Breath",
    "Respiratory_discomfort": "Respiratory Discomfort",
    "SPO2": "Oxygen Saturation (SpO2%)",
    "Dihareea": "Diarrhea",  # Corrected spelling
    "Vomitting": "Vomiting",  # Corrected spelling
    "Cardiovascular": "Cardiovascular Disease",
    "Asthma": "Asthma",
    "Diabetis": "Diabetes",  # Corrected spelling
    "Pulmonary": "Pulmonary Disease",
    "Immunosuppresion": "Immunosuppression",  # Corrected spelling
    "Obesity": "Obesity",
    "Liver": "Liver Disease",
    "Neurologic": "Neurological Disorder",
    "Renal": "Renal Disease",
}

DICTIONARY_TO_CLINICAL_NAMES_CUTRACT = {
    "age": "Age",
    "psa": "Prostate-Specific Antigen Level",
    "mortCancer": "Mortality Due to Cancer",
    "mort": "Overall Mortality",
    "days": "Days from Diagnosis to Event (e.g., Death or Last Follow-Up)",
    "comorbidities": "Comorbid Conditions (e.g., Diabetes, Hypertension)",
    "treatment": "Treatment Received (e.g., Surgery, Radiation, Chemotherapy)",
    "grade": "Tumor Grade (e.g., Gleason Score for Prostate Cancer)",
    "stage": "Cancer Stage (e.g., TNM Staging System)",
    "gleason1": "Primary Gleason Pattern Score",
    "gleason2": "Secondary Gleason Pattern Score",
}

DICTIONARY_TO_CLINICAL_NAMES_SEER = {
    "age": "Age (years)",
    "psa": "Prostate-Specific Antigen (PSA) level",
    "mortCancer": "Cancer-specific mortality",
    "mort": "Overall mortality",
    "days": "Days from diagnosis to last follow-up or death",
    "comorbidities": "Number of comorbidities",
    "treatment": "Treatment received (e.g., surgery, radiation, etc.)",
    "grade": "Tumor grade",
    "stage": "Cancer stage (TNM classification)",
    "gleason1": "Primary Gleason grade",
    "gleason2": "Secondary Gleason grade",
}
