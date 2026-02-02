# TDSE_HeartDiseasePrediction_LogisticRegression

**Created by**
Juan Carlos Leal Cruz 

## Laboratory Description
The following laboratory applies logistic regression to predict heart disease risk using the UCI Heart Disease Dataset, which contains 303 patient records with 14 clinical features. The objective is to classify the presence or absence of heart disease by implementing logistic regression from scratch using NumPy, including the sigmoid function, cost computation, and gradient descent. The lab emphasizes model interpretation through parameter tuning, decision boundary visualization, regularization, and concludes with a brief exploration of deployment concepts using Amazon SageMaker.

### Prerequisites

To run this laboratory you will need:

- Python 3.8 or higher (As a recomendation, the newer versions are better due to the compatibility with the creations of virtual environments in multiple IDEs)
- Jupyter Notebook or JupyterLab
- The following Python libraries:
  - ``numpy``
  - ``matplotlib``
  - ``pandas``
 
### Execution
To run this laboratory, follow the steps below:

1. Clone the repository and navigate to the folder:
   ```
   git clone <Repository_URL>
   cd <Repository_name>
   ```

2. Setup the virtual environment
   ```
   python -m venv venv
   source venv/bin/activate       # On Linux/Mac
   venv\Scripts\activate          # On Windows
   ```

3. Start running each block of code so you can see the results
___

### Laboratory Summary
- Load and explore the dataset (EDA, preprocessing, train/test split, normalization).  
- Implement logistic regression from scratch (sigmoid, cost, gradient descent).  
- Visualize decision boundaries for selected feature pairs.  
- Apply L2 regularization and tune λ.  
- Deploy trained model on Amazon SageMaker for real-time inference.

### Dataset Description
- **Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease)  
- **Size:** 271 patients  
- **Features:** Age (29–77), Cholesterol (112–564 mg/dL), Resting BP, Max HR, ST Depression, Vessels, and others  
- **Target:** Heart disease presence (~55% positive)  
- **Notes:** Binarized target (1 = disease, 0 = no disease). Selected ≥6 features for modeling (Age, Cholesterol, BP, Max HR, ST Depression, Vessels).

---

### Laboratory Steps

This project is divided into five distinct steps, moving from data preparation to cloud deployment.

#### Step 1: Load and Prepare the Dataset
* **Data Acquisition:** Downloaded `heart.csv` from Kaggle.
* **Preprocessing:** Binarized the target column (1=disease, 0=absence), performed a stratified 70/30 train/test split, and normalized numerical features.
* **Exploratory Data Analysis (EDA):** Summarized statistics, handled outliers, and visualized class distributions.

#### Step 2: Implement Basic Logistic Regression
* **Core Functions:** Implemented `sigmoid`, `cost_function` (binary cross-entropy), and `gradient_descent` using NumPy.
* **Training:** Trained the model on the full training set ($\alpha \approx 0.01$, 1000+ iterations).
* **Evaluation:** Calculated Accuracy, Precision, Recall, and F1 scores on both training and test sets.

#### Step 3: Visualize Decision Boundaries
* **Feature Selection:** Selected specific feature pairs (e.g., Age vs. Cholesterol, BP vs. Max HR).
* **Visualization:** Sub-setted data to 2D, trained specific models for these pairs, and plotted the decision boundary lines against scatter plots of true labels to analyze separability.

#### Step 4: Regularization (L2)
* **Implementation:** Added L2 Regularization to the cost function and gradient updates to prevent overfitting.
* **Tuning:** Tuned the regularization parameter $\lambda$ (values: `[0, 0.001, 0.01, 0.1, 1]`).
* **Analysis:** Compared decision boundaries and metrics between un-regularized and regularized models.

#### Step 5: Deployment Evidence (Amazon SageMaker)
The final model was exported and deployed to an endpoint using Amazon SageMaker.
