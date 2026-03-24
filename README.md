🏥 Insurance Charges Predictor

A novel segmented machine learning approach to insurance cost prediction — going beyond the standard Kaggle polynomial regression solution.




🔗 Live Demo
👉 Try the model here: YOUR_LIVE_LINK_HERE
Enter your age, BMI, smoking status, and other details — get an instant insurance charge prediction.

💼 Business Problem
Insurance companies face a fundamental pricing challenge — charge too much and lose customers, charge too little and incur losses. Accurate charge prediction directly impacts profitability, competitive pricing, and risk management.
The real-world complexity is that insurance costs are not uniform. A 55-year-old obese smoker represents an entirely different risk profile than a 25-year-old non-smoker with low BMI. A single model trying to learn both groups simultaneously splits its capacity and underserves both.
This project solves that by treating them as what they are — two fundamentally different populations.
Business Value This Model Delivers

Risk-based pricing — insurers can price policies more accurately per risk segment
Actuarial support — data-driven charge estimation reduces reliance on manual actuarial tables
Customer transparency — live app lets individuals understand what drives their insurance cost
Fraud detection groundwork — large deviations between predicted and actual charges can flag anomalies


📌 My Approach vs Everyone Else on Kaggle
Most Kaggle solutions for this dataset follow the same path:
Load data → Encode categoricals → Apply Polynomial Regression → Report R²
That is a valid starting point. But it treats the entire dataset as one homogeneous population — which it is not.
My Thinking Process
When I explored the data, I noticed something others overlooked:

Smokers and non-smokers do not just have different average charges — they follow different mathematical relationships entirely.

A non-smoker's charges are driven primarily by age and BMI in a relatively smooth curve. A smoker's charges spike sharply and interact with BMI and age in a compounding, non-linear way that no polynomial term fully captures.
This is not a subtle pattern. It is a structural break in the data — the kind of domain insight that separates a data scientist from someone who just runs algorithms.
Why Segmented RF, Not Polynomial?
The standard comparison on this dataset is linear regression vs polynomial regression. I skipped that comparison entirely because it was the wrong question.
What I askedWhat most askWhich algorithm best fits each risk group?Does polynomial beat linear?Does the data have structural segments?What degree polynomial works best?What would a business actually deploy?What gets the best Kaggle score?
Polynomial regression — even with degree=2 — still fits one curve to all 1,338 records. It cannot know that the smoking/non-smoking divide means the data has two fundamentally different distributions sitting on top of each other.
Random Forest, by contrast:

Makes no assumption about the shape of the relationship
Naturally captures non-linear interactions (BMI × smoking, age × BMI)
Can be dedicated to each segment, learning each group's patterns without interference from the other
Is robust to outliers — important because high-cost non-smokers are rare and extreme

The result: Segmented RF beats polynomial on every single metric — smoker segment, non-smoker segment, and combined.

📊 Model Performance
SegmentTraditional PolynomialSegmented RFImprovementSmoker R²0.78100.8290+6.14%Non-Smoker R²0.44440.4612+3.77%Combined R²0.86780.8783+1.22%Combined RMSE$4,530$4,345$185 less error per prediction

Non-smoker R² of 0.46 is a known data ceiling. Every model tested — RF, Polynomial, XGBoost, Hybrid — converged around the same score. This confirms the ceiling is in the data (missing health history, pre-existing conditions), not the algorithm. Knowing this distinction is itself part of good data science.


🧪 Exhaustive Model Testing
I did not pick Segmented RF because it sounded good. I tested every reasonable alternative and let the numbers decide:
ApproachCombined R²VerdictSegmented RF0.8783✅ Final model — best on all metricsHybrid (Smoker RF + Non-Smoker Poly)0.8750❌ Non-smoker polynomial worse on its own segmentXGBoost unified0.8710❌ Segmentation advantage beats algorithm advantagePolynomial baseline0.8678Baseline — what most Kaggle solutions stop atSegmented RF + sample weights0.8678❌ Weights hurt majority-class non-smoker predictions
Each failed approach taught something concrete:

XGBoost losing proves that knowing your data structure matters more than using a fancier algorithm
Sample weights failing proves that RF already handles imbalance internally — adding manual weights was redundant
Non-smoker polynomial failing proves the polynomial fit breaks down without the smoker population anchoring it


🧠 Technical Depth
Feature Engineering
Engineered 11 additional features from the original 6 to capture non-linear relationships and domain knowledge:
pythonage²                    # insurance costs accelerate with age, not linear
bmi²                    # obesity risk compounds non-linearly
age × bmi               # interaction — older + heavier = compounding risk
bmi_obese               # risk zone flag (BMI ≥ 30)
bmi_severely_obese      # risk zone flag (BMI ≥ 35)
bmi × bmi_obese         # obese interaction term
age_group               # age buckets — risk jumps at life stages, not smoothly
age × children          # family size risk scales with age
smoker × bmi            # smoking + obesity = highest risk combination
smoker × age            # smoking compounds with age non-linearly
smoker × bmi_obese      # triple risk factor interaction
Evaluation Methodology

Single train/test split (80/20) done once — both models see the same 268 test rows
5-fold cross-validation on both RF models to check stability, not just peak performance
Per-segment reporting — smoker and non-smoker evaluated separately AND combined
Traditional polynomial evaluated on the same test segments for a true apples-to-apples comparison

Pipeline Design
All preprocessing inside scikit-learn Pipelines — zero data leakage between training and test:
python# Each segment gets its own dedicated preprocessor + model
rf_smoker_preprocessor    → StandardScaler + OneHotEncoder → RF Smoker
rf_nonsmoker_preprocessor → StandardScaler + OneHotEncoder → RF Non-Smoker

🏗️ Architecture
Input (6 raw features)
         │
         ▼
Feature Engineering (→ 17 features)
         │
         ▼
   Smoker? ──── yes ──► Preprocessor ──► RF Smoker Model     ──► Prediction
         │                                   (R² = 0.83)
         └────── no  ──► Preprocessor ──► RF Non-Smoker Model ──► Prediction
                                              (R² = 0.46)

🗂️ Dataset
FeatureTypeDescriptionageNumericalAge of beneficiary (18–64)sexCategoricalmale / femalebmiNumericalBody Mass IndexchildrenNumericalNumber of dependantssmokerCategoricalyes / noregionCategoricalnortheast / northwest / southeast / southwestchargesTargetAnnual insurance cost in USD
1,338 rows — no missing values — benchmark insurance dataset

📁 Project Structure
insurance-charges-predictor/
├── insurance.csv                   # Dataset (1,338 rows)
├── insurance_model.py              # Training — Segmented RF vs Polynomial comparison
├── app.py                          # Streamlit live app
├── model_smoker.joblib             # Trained smoker RF model
├── model_nonsmoker.joblib          # Trained non-smoker RF model
├── preprocessor_smoker.joblib      # Smoker preprocessor
├── preprocessor_nonsmoker.joblib   # Non-smoker preprocessor
├── model_traditional.joblib        # Polynomial baseline (for comparison)
├── requirements.txt                # Dependencies
└── README.md                       # This file

🔍 Honest Limitations

Non-smoker R² = 0.46 — missing features (health history, pre-existing conditions, claims history) are the real drivers of non-smoker cost variation. This is a data ceiling, not a modelling failure — confirmed by every algorithm converging at the same score.
Static dataset — production deployment would require live data pipelines and scheduled retraining.
Small smoker training set — 220 rows for the smoker model causes high CV variance (±0.078). More data would stabilise it.


🔮 Future Improvements

 Add health history features to push non-smoker R² past 0.60
 Implement segmented XGBoost as direct comparison to segmented RF
 Add SHAP values for individual prediction explainability
 Confidence intervals on predictions for risk-based decision making
 Retraining pipeline for production drift handling


👤 Author
Dua e Fatima



Built with Python & scikit-learn — going beyond the standard Kaggle solution
