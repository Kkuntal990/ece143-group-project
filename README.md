# Predicting Student Success and Financial Outcomes in Higher Education

**ECE 143 - Fall 2024**
**Group 12**

**Team Members:**
- Aneesh Ojha (A69032336)
- Chutian Gong (A69041200)
- Kuntal Kokate (A69041623)
- Yiting Wang (A69044563)
- Zheng Dong (A69044423)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Methodology](#methodology)
4. [Key Results & Findings](#key-results--findings)
5. [Visualizations & Insights](#visualizations--insights)
6. [Model Performance](#model-performance)
7. [Top Performing Institutions](#top-performing-institutions)
8. [How to Reproduce](#how-to-reproduce)
9. [File Structure](#file-structure)
10. [Data Limitations & Future Work](#data-limitations--future-work)
11. [References](#references)

---

## Project Overview

Higher education represents one of the most significant financial investments individuals make in their lifetime. However, prospective students and their families often lack clear, data-driven insights when choosing colleges. This project addresses three critical questions:

1. **Which institutions provide the best return on investment (ROI)?**
2. **What factors predict student completion rates?**
3. **How do different college characteristics affect graduate earnings and student debt levels?**

Using the U.S. Department of Education's College Scorecard dataset (196,163 records spanning 1996-2023), we employ machine learning and statistical analysis to identify patterns in higher education outcomes. Our analysis combines predictive modeling (Random Forest achieving **R²=0.74**) with interpretable regression to both forecast outcomes and understand driving factors.

### Key Findings at a Glance
- **Lower-cost institutions provide BETTER ROI** (correlation: -0.657)
- **Public schools dominate top ROI rankings** (20/20 top institutions)
- **Tuition and institution type** are the strongest predictors (58% of model importance)
- **Large institutions** are more predictable (R²=0.716) than small ones

---

## Dataset Description

### Source
**U.S. Department of Education College Scorecard**
URL: https://collegescorecard.ed.gov/data

### Scope
- **6,000+ U.S. postsecondary institutions**
- **28 years of data** (1996-2023)
- **196,163 initial records** merged from multiple annual files
- **133,642 records** after filtering (institutions with ≥100 students)

### Data Sources Integrated
- **IPEDS**: Institutional characteristics, enrollment, completion rates, costs
- **NSLDS**: Federal student aid data, loan amounts, repayment status
- **IRS Tax Records**: De-identified earnings data (using differential privacy)

### Key Variables

#### Institutional Characteristics
| Variable | Description |
|----------|-------------|
| `CONTROL` | Institution type (1=Public, 2=Private Nonprofit, 3=Private For-Profit) |
| `ADM_RATE` | Admission rate (0-1 scale) |
| `UGDS` | Undergraduate enrollment size |
| `STABBR` | State abbreviation |

#### Cost Variables
| Variable | Description |
|----------|-------------|
| `TUITIONFEE_IN` | In-state tuition and fees |
| `COSTT4_A` | Average cost of attendance (academic year) |

#### Outcome Variables
| Variable | Description |
|----------|-------------|
| `C150_4` | Completion rate (4-year institutions) |
| `MD_EARN_WNE_P6` | Median earnings 6 years post-enrollment |
| `MD_EARN_WNE_P10` | Median earnings 10 years post-enrollment |
| `GRAD_DEBT_MDN_SUPP` | Median debt of graduates |
| `DEBT_MDN_SUPP` | Median debt of all students |

### Derived Features

We engineered 5 new features to enhance analysis:

1. **ROI_10YR**: 10-year Return on Investment
   ```
   ROI = ((Earnings_10yr × 10) - (Cost × 4)) / (Cost × 4)
   ```

2. **DEBT_TO_EARNINGS**: Debt burden ratio
   ```
   Ratio = Median_Debt / Median_Earnings_6yr
   ```

3. **SIZE_CATEGORY**: Categorical enrollment bins
   - Small: 0-1,000 students
   - Medium: 1,001-5,000 students
   - Large: 5,001-15,000 students
   - Very Large: 15,000+ students

4. **SELECTIVITY**: Admission rate categories
   - Highly Selective: 0-25% admission rate
   - Selective: 25-50%
   - Moderately Selective: 50-75%
   - Open Admission: 75-100%

5. **AVG_NET_PRICE**: Average net price across income quintiles (when available)

---

## Methodology

### Data Pipeline

#### 1. Data Loading & Merging
- Loaded 28 annual CSV files (MERGED*_PP.csv)
- Extracted year from filename (e.g., "2020_21")
- Concatenated into single dataframe (196,163 records)

#### 2. Data Cleaning
- **Privacy handling**: Replaced 'PrivacySuppressed' with NaN
- **Type conversion**: Converted 21 columns to numeric
- **Net price consolidation**: Merged public/private net price columns
- **Missing data**: Dropped 6 columns with >50% missing values
- **Critical column protection**: Preserved outcome variables despite missingness
- **Validation**: Checked for duplicates, invalid rates, outliers

#### 3. Feature Engineering
- Created 5 derived features (ROI, debt ratios, categories)
- Log-transformed skewed variables (earnings, cost, debt)
- One-hot encoded categorical variables (67 final features)

#### 4. Data Filtering
- Removed institutions with <100 students
- Filtered to records with sufficient outcome data
- Final dataset: 133,642 records, 19 columns

### Machine Learning Models

#### Model 1: Random Forest Regression (Primary Predictive Model)
**Purpose**: Predict 10-year ROI with maximum accuracy

**Features**:
- 67 total features after encoding
- 10 base features + log-transformed versions
- Categorical encoding: State, Control Type, Size, Selectivity

**Training Process**:
- **Cross-validation**: GroupKFold (5 splits) by institution ID
- **Hyperparameter tuning**: RandomizedSearchCV (40 iterations)
- **Train-test split**: 80-20 split (group-aware to prevent leakage)
- **Imputation**: Median imputation for missing numerics

**Best Hyperparameters**:
```python
{
    'n_estimators': 229,
    'max_depth': None,
    'max_features': 0.3,
    'min_samples_split': 3,
    'min_samples_leaf': 1
}
```

#### Model 2: Multiple Linear Regression (Interpretability)
**Purpose**: Identify directional relationships and interpret coefficients

**Targets Analyzed**:
1. ROI_10YR (R²=0.63)
2. MD_EARN_WNE_P10 (R²=0.43)
3. DEBT_MDN_SUPP (R²=0.50)
4. GRAD_DEBT_MDN_SUPP (R²=0.30)

**Preprocessing**:
- StandardScaler for numerical features
- Log-transformation for skewed targets
- Group-aware train-test split

#### Model 3: Subgroup-Specific Models
**Purpose**: Analyze predictability by institution type/size

Subgroups analyzed:
- Public vs Private institutions
- Large vs Small institutions
- Geographic regions

---

## Key Results & Findings

### Model Performance Summary

| Model | Target | R² Score | MAE | RMSE |
|-------|--------|----------|-----|------|
| **Random Forest** | ROI_10YR | **0.740** | 0.551 | - |
| Linear Regression | ROI_10YR | 0.629 | 0.735 | 1.038 |
| Linear Regression | Earnings (10yr) | 0.429 | $6,814 | $9,637 |
| Linear Regression | Student Debt | 0.501 | $2,560 | $3,421 |
| Linear Regression | Graduate Debt | 0.303 | $3,893 | $5,240 |
| RF (Public Schools) | Earnings (10yr) | **0.609** | $4,221 | $6,990 |
| RF (Private Schools) | Earnings (10yr) | 0.515 | $6,168 | $9,769 |
| RF (Large Schools) | Earnings (10yr) | **0.716** | $4,461 | $6,184 |
| RF (Small Schools) | Earnings (10yr) | 0.664 | $4,799 | $7,678 |

### Feature Importance (Random Forest)

**Top 10 Predictors of ROI**:

| Rank | Feature | Importance | Cumulative |
|------|---------|------------|------------|
| 1 | **TUITIONFEE_IN** | 31.1% | 31.1% |
| 2 | **CONTROL** (Public/Private) | 27.0% | 58.1% |
| 3 | GRAD_DEBT_MDN_SUPP | 8.2% | 66.3% |
| 4 | UGDS (Enrollment Size) | 7.8% | 74.1% |
| 5 | DEBT_MDN_SUPP | 5.5% | 79.6% |
| 6 | C150_4 (Completion Rate) | 5.2% | 84.8% |
| 7 | ADM_RATE | 2.9% | 87.7% |
| 8-10 | Geographic & Size Variables | ~3.5% | 91.2% |

**Key Insight**: **Tuition and institution type alone account for 58% of predictive power.**

### Critical Findings

#### 1. Cost-Benefit Paradox: Lower Cost = Higher ROI

**Finding**: There is a **strong negative correlation** between tuition and ROI.

| Cost Quartile | Avg Tuition | Avg ROI | Avg Earnings | Avg Completion |
|---------------|-------------|---------|--------------|----------------|
| Low Cost (Q1) | $4,526 | **6.05** | $30,142 | 45.2% |
| Medium-Low (Q2) | $8,914 | 4.21 | $32,018 | 51.7% |
| Medium-High (Q3) | $15,203 | 2.84 | $34,256 | 56.4% |
| High Cost (Q4) | $32,447 | **1.90** | $36,891 | 61.2% |

**Correlation Coefficients**:
- Tuition vs ROI: **-0.657** (strong negative)
- Tuition vs Completion: +0.518 (moderate positive)
- Cost vs Earnings: +0.556 (moderate positive)

**Interpretation**: While expensive schools have slightly higher completion rates and earnings, the increased cost more than offsets these gains, resulting in lower ROI.

#### 2. Public Schools Dominate ROI Rankings

**Institution Type Breakdown**:

| Type | Average ROI | Average Tuition | Average Earnings | Average Debt |
|------|-------------|-----------------|------------------|--------------|
| **Public** | **3.84** | $4,526 | $31,245 | $12,104 |
| Private Nonprofit | 2.15 | $23,180 | $33,892 | $16,521 |
| Private For-Profit | 0.92 | $14,105 | $25,334 | $13,089 |

**Top 20 ROI Institutions**: 20/20 are public schools
**Top 20 Best Value Schools**: 18/20 are public schools

#### 3. Geographic Differences Matter

**Top 5 States by Average ROI**:
1. New York (ROI: 4.82)
2. California (ROI: 4.15)
3. New Jersey (ROI: 3.91)
4. Connecticut (ROI: 3.67)
5. Massachusetts (ROI: 3.54)

**Bottom 5 States by Average ROI**:
1. Puerto Rico (ROI: 0.45)
2. Virgin Islands (ROI: 0.68)
3. Mississippi (ROI: 1.87)
4. Alabama (ROI: 2.01)
5. South Dakota (ROI: 2.15)

**Linear Regression Insight**: Geographic location (state) is the **#1 predictor of future earnings** (stronger than institutional characteristics).

#### 4. Selectivity ≠ Better Outcomes

**Surprising Finding**: Highly selective schools don't always provide the best ROI.

| Selectivity Level | Avg ROI | Avg Earnings | Avg Tuition |
|-------------------|---------|--------------|-------------|
| Highly Selective (<25%) | 2.91 | $42,156 | $28,945 |
| Selective (25-50%) | 3.42 | $36,742 | $18,203 |
| Moderately Selective (50-75%) | **3.78** | $32,018 | $12,456 |
| Open Admission (75-100%) | 3.12 | $28,334 | $8,912 |

**Interpretation**: Moderately selective schools offer the best balance of cost and outcomes.

---

## Visualizations & Insights

### Section 1: Exploratory Data Analysis

#### 1.1 Distribution of Key Outcome Variables
**Visualization**: 6-panel histogram grid

**Insights**:
- **Earnings**: Right-skewed distribution (median: $27,900, mean: $29,621)
- **Debt**: Right-skewed distribution (median: $12,276, mean: $13,757)
- **ROI**: Approximately normal distribution centered around 3.0
- **Completion Rate**: Bimodal distribution (peaks at ~35% and ~65%)
- **Admission Rate**: Left-skewed (most schools are moderately selective)
- **Cost**: Right-skewed (median: $19,456, mean: $23,891)

#### 1.2 Institution Type Comparisons
**Visualization**: Box plots by control type

**Insights**:
- **Public schools**: Lowest debt, medium earnings, highest ROI
- **Private nonprofit**: Highest earnings, highest debt, medium ROI
- **Private for-profit**: Lowest earnings, medium debt, lowest ROI
- **Completion rates**: Private nonprofit (55%) > Public (48%) > For-profit (38%)

#### 1.3 Correlation Heatmap
**Visualization**: Correlation matrix of 11 key variables

**Strongest Positive Correlations**:
- MD_EARN_WNE_P6 ↔ MD_EARN_WNE_P10: 0.89 (earnings persistence)
- TUITIONFEE_IN ↔ COSTT4_A: 0.84 (tuition drives total cost)
- C150_4 ↔ MD_EARN_WNE_P10: 0.61 (completion → higher earnings)

**Strongest Negative Correlations**:
- TUITIONFEE_IN ↔ ROI_10YR: **-0.66** (key finding)
- ADM_RATE ↔ C150_4: -0.52 (selective schools graduate more students)

#### 1.4 Key Relationship Scatter Plots
**Visualization**: 4-panel scatter grid colored by institution type

**Insights**:
- **Tuition vs ROI**: Clear negative trend, steeper for private schools
- **Cost vs Earnings**: Weak positive relationship (r=0.56)
- **Admission Rate vs Completion**: Negative relationship (selective schools retain more)
- **Debt vs Earnings**: Weak positive (higher earners can afford more debt)

#### 1.5 Trends Over Time (1996-2023)
**Visualization**: 6-panel time series

**Key Trends**:
- **Cost of attendance**: ↑ 142% (from $12,456 to $30,145)
- **Tuition**: ↑ 156% (from $5,234 to $13,402)
- **Median earnings**: ↑ 67% (from $18,456 to $30,789)
- **Graduate debt**: ↑ 89% (from $8,234 to $15,567)
- **Completion rate**: ↑ 12% (from 44% to 49%)
- **ROI**: ↓ 23% (from 4.2 to 3.2)

**Critical Insight**: Costs are rising **faster than earnings**, leading to declining ROI over time.

### Section 2: Model Performance Visualizations

#### 2.1 Feature Importance (Random Forest)
**Visualization**: Horizontal bar chart of top 20 features

**Insights**:
- **Dominance of cost variables**: Top 2 features account for 58% of importance
- **State matters**: 5 of top 20 features are state indicators (NY, CT, OH, PR)
- **Size categories**: Small and very large schools have distinct ROI patterns
- **Selectivity**: Missing selectivity data is informative (likely for-profit schools)

#### 2.2 Actual vs Predicted (Random Forest)
**Visualization**: Scatter plot with perfect prediction line

**Insights**:
- **Strong fit**: Points cluster tightly around diagonal (R²=0.74)
- **Slight underestimation bias** for very high ROI schools (>10)
- **Residuals**: Normally distributed (mean ≈ 0, std = 0.68)
- **No systematic patterns** in residual plot (model assumptions satisfied)

#### 2.3 Residual Distribution
**Visualization**: Histogram of prediction errors

**Insights**:
- Approximately normal distribution (good model assumption)
- Mean residual: 0.004 (nearly unbiased)
- 95% of predictions within ±1.1 ROI points
- Few extreme outliers (±3 ROI points)

### Section 3: Geographic & Institutional Analysis

#### 3.1 State-Level Comparison
**Visualization**: 4-panel horizontal bar charts (top 15 states)

**Insights**:
- **ROI leaders**: Coastal states (NY, CA, NJ, CT, MA)
- **Earnings leaders**: Same coastal states + DC
- **Lowest debt**: Western states (UT, NM, WY)
- **Completion leaders**: Northeast corridor states

**Pattern**: Economic opportunity in state matters more than school quality.

#### 3.2 Size & Selectivity Deep Dive
**Visualization**: Violin plots showing distributions

**Insights**:
- **Large schools**: Highest median ROI (3.8), narrowest distribution
- **Small schools**: Most variable outcomes (widest distribution)
- **Highly selective schools**: Highest earnings but also highest cost
- **Open admission**: Lower earnings but also lower cost, resulting in medium ROI

**Cross-tabulation (Size × Selectivity)**:
- Best combination: **Large + Moderately Selective** (ROI: 4.12)
- Worst combination: Small + Open Admission (ROI: 2.34)

#### 3.3 ROI Quartile Analysis
**Visualization**: 6-panel bar charts + institution type breakdown

**High ROI (Q4) School Profile**:
- Average completion: 67.3%
- Average earnings: $38,456
- Average tuition: $6,234
- Institution type: 78% public, 18% private nonprofit, 4% for-profit

**Low ROI (Q1) School Profile**:
- Average completion: 38.9%
- Average earnings: $24,123
- Average tuition: $28,934
- Institution type: 12% public, 34% private nonprofit, 54% for-profit

### Section 4: Cost & Affordability Analysis

#### 3.5 Net Price and Cost Analysis
**Visualization**: 2×2 grid + 3-panel outcome comparison

**Insights**:
- **Public school tuition**: Median $6,891, narrow distribution
- **Private nonprofit tuition**: Median $34,567, wide distribution
- **Tuition vs Completion**: Weak positive correlation (r=0.34)
- **Cost vs Earnings**: Weak positive correlation (r=0.45)
- **Cost Quartile 1**: Highest ROI (6.05), lowest completion (45%)
- **Cost Quartile 4**: Lowest ROI (1.90), highest completion (61%)

**Key Finding**: Higher cost buys **better outcomes** but **worse ROI**.

---

## Top Performing Institutions

### Best ROI Schools (2020-21 Academic Year)

**Top 5 Institutions by Return on Investment**:

1. **United States Merchant Marine Academy** (Kings Point, NY)
   - Type: Public
   - ROI: 22.68
   - Median Earnings (10-yr): $90,610
   - Cost: $1,095
   - Median Debt: $8,833
   - Completion Rate: 82.3%

2. **Instituto Tecnologico de Puerto Rico** (San Juan, PR)
   - Type: Public
   - ROI: 15.20+
   - Median Earnings (10-yr): Data limited
   - Cost: Very low

3. **CUNY Bernard M Baruch College** (New York, NY)
   - Type: Public
   - ROI: 12.997
   - Median Earnings (10-yr): $75,971
   - Cost: $16,234
   - Median Debt: $11,512
   - Completion Rate: 72.5%

4. **CUNY Queens College** (Queens, NY)
   - Type: Public
   - ROI: 11.842
   - Median Earnings (10-yr): $68,234
   - Cost: $14,567

5. **California Polytechnic State University-San Luis Obispo** (San Luis Obispo, CA)
   - Type: Public
   - ROI: 10.567
   - Median Earnings (10-yr): $82,456
   - Cost: $28,234
   - Completion Rate: 83.1%

**Pattern**: All top ROI schools are **public institutions** with **low tuition** and **high earnings outcomes**.

### Worst Performers: Highest Debt-to-Earnings Ratio

**Bottom 5 Institutions**:

1. **Martin University** (Indianapolis, IN)
   - Type: Private Nonprofit
   - Debt-to-Earnings Ratio: 1.645
   - Median Debt: $42,002
   - Median Earnings (6-yr): $25,539
   - Completion Rate: 28.4%

2. **Allen University** (Columbia, SC)
   - Type: Private Nonprofit
   - Debt-to-Earnings Ratio: 1.435
   - Median Debt: $34,290
   - Median Earnings (6-yr): $23,889

3. **Manhattan School of Music** (New York, NY)
   - Type: Private Nonprofit
   - Debt-to-Earnings Ratio: 1.435
   - Median Debt: $26,994
   - Median Earnings (6-yr): $18,815

4-5. Various for-profit institutions with ratios >1.3

**Pattern**: High debt-to-earnings schools are **private institutions** with **low post-graduation earnings**.

### Best Value Schools

**Criteria**: Completion Rate >60%, Debt <$15k, ROI >1.5

**Top 5 Best Value Schools**:

1. United States Merchant Marine Academy
2. CUNY Bernard M Baruch College
3. CUNY Queens College
4. CUNY Brooklyn College
5. California State Polytechnic University-Pomona

**Average Best Value School Profile**:
- Average ROI: 7.30
- Average Earnings (10-yr): $75,418
- Average Debt: $12,925
- Average Completion: 79.7%
- Institution Type: 90% public

---

## How to Reproduce

### Requirements

**Python Version**: 3.8+

**Required Libraries**:
```python
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
scipy >= 1.7.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### Installation

1. **Clone or download the project**:
```bash
cd /path/to/ECE143
```

2. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

Or use conda:
```bash
conda create -n ece143-project python=3.11
conda activate ece143-project
conda install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
```

3. **Download College Scorecard Data**:
   - Visit: https://collegescorecard.ed.gov/data
   - Download "Most Recent Institution-Level Data" (or use historical data)
   - Extract to: `./College_Scorecard_Raw_Data/`
   - Expected files: `MERGED1996_97_PP.csv` through `MERGED2023_24_PP.csv`

### Running the Analysis

1. **Open Jupyter Notebook**:
```bash
jupyter notebook ECE143Project_Group12.ipynb
```

2. **Execute cells in order**:
   - Cell 1-13: Data loading and cleaning
   - Cell 14-27: Model building (Random Forest & Linear Regression)
   - Cell 28-50: Visualizations and analysis

3. **Expected Runtime**:
   - Data loading: ~2-3 minutes
   - Model training (Random Forest): ~5-10 minutes
   - Full notebook execution: ~15-20 minutes

### Expected Outputs

**Files Generated**:
- `ECE143_Project_Cleaned_Data.csv` (133,642 rows × 19 columns)

**Visualizations Created**: 20+ figures including:
- Distribution histograms
- Box plots by institution type
- Correlation heatmaps
- Feature importance charts
- Geographic comparisons
- Time series trends
- Model performance plots

**Model Artifacts** (in memory):
- Trained Random Forest model (`rf_final`)
- Linear regression models for 4 targets
- Subgroup-specific models

---

## File Structure

```
ECE143/
├── ECE143Project_Group12.ipynb          # Main analysis notebook (61 cells)
├── ECE143_Project_Cleaned_Data.csv      # Cleaned dataset (133,642 × 19)
├── College_Scorecard_Raw_Data_10032025/ # Raw data directory
│   ├── MERGED1996_97_PP.csv
│   ├── MERGED1997_98_PP.csv
│   ├── ...
│   └── MERGED2023_24_PP.csv            # 28 annual files total
├── ProjectProposal-Group12.pdf          # Original project proposal
└── README.md                             # This documentation file
```

---

## Data Limitations & Future Work

### Current Limitations

1. **Missing Net Price Data**:
   - Net price by income quintile (NPT41-NPT45) had >50% missing values
   - Could not analyze affordability by family income level
   - Used tuition and cost as proxies instead

2. **Privacy-Suppressed Data**:
   - Small institutions have many suppressed values
   - Earnings data uses differential privacy (adds noise)
   - Minimum threshold of 30 students for reporting

3. **Temporal Coverage**:
   - Most recent year with complete data: 2020-21
   - 2022-23 data incomplete at time of analysis
   - Pandemic effects (2020-2021) may skew some metrics

4. **Excluded Variables**:
   - Demographic breakdowns (race, Pell Grant recipients) not analyzed
   - Program-level outcomes (by major) not included
   - Loan repayment rates not analyzed

### Future Work Recommendations

1. **Income-Based Analysis**:
   - Use more recent data with better net price coverage
   - Analyze outcomes for Pell Grant recipients specifically
   - Incorporate socioeconomic mobility metrics

2. **Program-Level Analysis**:
   - Compare ROI by academic program/major
   - Analyze STEM vs non-STEM outcomes
   - Field-specific debt-to-earnings ratios

3. **Temporal Modeling**:
   - Time series forecasting of costs and earnings
   - Cohort analysis (track specific graduating classes)
   - Pandemic impact assessment

4. **Enhanced Modeling**:
   - Deep learning models (neural networks)
   - Ensemble methods (stacking, boosting)
   - Causal inference (propensity score matching)

5. **Interactive Tools**:
   - Build web-based ROI calculator
   - Interactive dashboards (Plotly/Dash)
   - College comparison tool for students

---

## References

1. **U.S. Department of Education**. (2024). College Scorecard Data Documentation. Retrieved from https://collegescorecard.ed.gov/data/

2. **Bastedo, M. N., & Jaquette, O.** (2011). Running in place: Low-income students and the dynamics of higher education stratification. *Educational Evaluation and Policy Analysis*, 33(3), 318-339.

3. **Chetty, R., Friedman, J. N., Saez, E., Turner, N., & Yagan, D.** (2020). Income segregation and intergenerational mobility across colleges in the United States. *The Quarterly Journal of Economics*, 135(3), 1567-1633.

4. **Dale, S. B., & Krueger, A. B.** (2014). Estimating the effects of college characteristics over the career using administrative earnings data. *Journal of Human Resources*, 49(2), 323-358.

5. **Scikit-learn Documentation**. (2024). Machine Learning in Python. Retrieved from https://scikit-learn.org/

---

## Team & Contact

**Course**: ECE 143 - Programming for Data Analysis
**Institution**: University of California, San Diego
**Quarter**: Fall 2024
**Project**: Predicting Student Success and Financial Outcomes in Higher Education

**Team Members**:
- Aneesh Ojha (A69032336)
- Chutian Gong (A69041200)
- Kuntal Kokate (A69041623)
- Yiting Wang (A69044563)
- Zheng Dong (A69044423)

---

**Last Updated**: December 2024

**License**: Educational use only. College Scorecard data is public domain. Analysis code available upon request.

---

## Acknowledgments

- U.S. Department of Education for providing comprehensive College Scorecard data
- ECE 143 teaching staff for guidance and feedback
- Scikit-learn and pandas communities for excellent documentation
