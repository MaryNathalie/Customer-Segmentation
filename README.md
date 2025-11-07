# Online Retail Customer Analytics

This project analyzes the [Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) from the UCI Machine Learning Repository. It performs a comprehensive analysis including data preparation, RFMT (Recency, Frequency, Monetary, Tenure) feature engineering, customer segmentation via K-Means clustering, customer retention analysis using cohorts, and product association analysis using the Apriori algorithm.

Goal: uncover actionable insights by identifying distinct customer segments (e.g., "Champions," "At-Risk," "Big-Spenders") and discovering which products are frequently purchased together.

## 1\. Project Overview

This repository is structured as a series of five Jupyter notebooks that walk through a complete data analysis pipeline:

1.  **Data Preparation (`01_data_preparation.ipynb`):** Loads the raw data, handles missing values (especially `CustomerID`), removes duplicates, cleans anomalous data (e.g., non-positive `Quantity` or `UnitPrice`), and filters out non-product transactions (e.g., 'POSTAGE', 'BANK CHARGES').
2.  **Feature Engineering (`02_feature_engineering.ipynb`):** Calculates `Monetary` value for transactions and then uses the `lifetimes` library to aggregate transactional data into a customer-level RFMT (Recency, Frequency, Monetary, Tenure) summary.
3.  **Clustering (`03_clustering_rfmt.ipynb`):** Pre-processes the RFMT data by handling outliers (using `IsolationForest`) and scaling (using `StandardScaler`). It then uses the K-Means algorithm to segment customers into distinct groups. The optimal cluster number (K=4) is determined using the Elbow Method, Silhouette Score, Calinski-Harabasz Score, and Davies-Bouldin Score.
4.  **Cohort Analysis (`04_analysis_cohort.ipynb`):** Analyzes customer retention by grouping customers into monthly acquisition cohorts. It tracks the percentage of customers from each cohort who return for repeat purchases in subsequent months.
5.  **Market Basket Analysis (`05_analysis_market-basket.ipynb`):** Cleans and standardizes product `Description` text. It then transforms the data into a transaction-item matrix and applies the Apriori algorithm to find frequent itemsets and generate association rules.

## 2\. Running the Application

### a. Prerequisites

The project relies on several key Python libraries. The main ones are:

  * `pandas` & `numpy` (for data manipulation)
  * `ucimlrepo` (for fetching the dataset)
  * `lifetimes` (for RFMT feature engineering)
  * `scikit-learn` (for StandardScaler, IsolationForest, KMeans, metrics)
  * `yellowbrick` (for KElbowVisualizer)
  * `mlxtend` (for Apriori and association rules)
  * `matplotlib` & `seaborn` (for visualization)

### b. Setup & Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**
    (As you mentioned, you will create a `requirements.txt`. The command would be:)

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the notebooks:**
    The notebooks are designed to be run in sequence, as each one typically generates a CSV file that serves as the input for the next.

      * `01_data_preparation.ipynb` -\> outputs `prepared_data.csv`
      * `02_feature_engineering.ipynb` -\> outputs `customer_rfmt_data.csv`, `customer_rfmt_final.csv`
      * `03_clustering_rfmt.ipynb` -\> uses `customer_rfmt_final.csv`
      * `04_analysis_cohort.ipynb` -\> uses `customer_rfmt_data.csv`
      * `05_analysis_market-basket.ipynb` -\> outputs `prepared_data_descriptions.csv` (which is then used in the same notebook)

## 3\. Dataset Summary

  * **Source:** UCI Machine Learning Repository, Online Retail Dataset.
  * **Content:** Transactional data from a UK-based, non-store online retail company.
  * **Timeframe:** 01/12/2010 to 09/12/2011.
  * **Original Size:** 541,909 rows and 8 columns.
  * **Cleaned Size:** 399,573 rows (after removing entries with no `CustomerID`, duplicates, and non-positive `UnitPrice`).
  * **Key Features:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

## 4\. Design Decisions

Several key decisions were made during the cleaning and modeling phases:

  * **Handling Missing CustomerIDs:** 24.93% of the data was missing a `CustomerID`. Since this ID is essential for all user-centric analyses (clustering, cohorts, RFMT), these rows were **dropped**.
  * **Text Cleaning for MBA:** The `Description` field was highly inconsistent. A cleaning pipeline was built to standardize text by converting to uppercase, removing trailing punctuation, and replacing common abbreviations (e.g., `&` with `AND`, `S/` with `SET OF`).
  * **Handling Inconsistencies:** Some `StockCode`s mapped to multiple `Description`s (and vice-versa). A resolution function was created to map all conflicting entries to the most frequent (modal) value for that group, ensuring consistency.
  * **Outlier Removal for Clustering:** `Frequency` and `Monetary` data were heavily skewed by extreme outliers. `IsolationForest` was used to identify and remove the top 3% of these outliers before clustering, which resulted in more distinct and meaningful cluster definitions.
  * **Cluster Model Selection:** K-Means was chosen for its simplicity and interpretability. Features were scaled with `StandardScaler`. While log-transformation was tested, evaluation metrics (Silhouette, Davies-Bouldin) showed that K=4 on the scaled, non-transformed data (with outliers removed) produced the most mathematically distinct clusters.

## 5\. Insights

### Customer Segmentation (K-Means)

The analysis identified four distinct customer segments:

  * **Cluster 2: Champions (295 customers)**

      * **Profile:** High tenure (348 days), highest recency (335 days), and by far the **highest frequency** (avg. 13 repeat purchases).
      * **Value:** Loyal, long-term customers with a high AOV (398).
      * **Action:** Reward loyalty, solicit reviews, offer exclusive access.

  * **Cluster 1: Big-Ticket Spenders (277 customers)**

      * **Profile:** Low frequency (2.9) but the **highest AOV (884)** by a large margin.
      * **Value:** They don't buy often, but spend significantly when they do.
      * **Action:** Market high-value items, offer "white glove" service, avoid high-frequency discount-based marketing.

  * **Cluster 0: Occasional / Established (1210 customers)**

      * **Profile:** High tenure (316 days) but low frequency (3.3) and the lowest AOV (288).
      * **Value:** They are long-time customers but are not highly engaged.
      * **Action:** Re-engage with personalized recommendations based on past purchases (see MBA).

  * **Cluster 3: New / At-Risk (899 customers)**

      * **Profile:** **Lowest tenure** (150 days), **lowest recency** (84 days), and **lowest frequency** (1.85).
      * **Value:** This group contains new customers and old customers who are churning (low recency).
      * **Action:** Target with welcome offers (for new users) or "we miss you" campaigns (for churning users).

### Retention (Cohort Analysis)

  * **Strongest Cohort:** The first cohort (December 2010) was the largest (768 users) and had the best long-term retention, with 57.4% returning in their 12th month (likely seasonal Christmas shoppers).
  * **Early Churn:** All cohorts show a significant drop in retention after the first month, indicating a critical need for better onboarding and first-month engagement strategies.
  * **Successful Acquisition Period:** The September and October 2011 cohorts showed very high initial retention (50-74% in months 2-3), suggesting a successful marketing or acquisition campaign during that time.

### Product Associations (Market Basket Analysis)

  * **Top Frequent Items:** The most commonly purchased items are `WHITE HANGING HEART T-LIGHT HOLDER`, `REGENCY CAKESTAND 3 TIER`, and `JUMBO BAG RED RETROSPOT`.
  * **Top Association Rules:**
    1.  Customers who buy `POPPY'S PLAYHOUSE KITCHEN` and `POPPY'S PLAYHOUSE LIVINGROOM` are highly likely to also buy `POPPY'S PLAYHOUSE BEDROOM`.
    2.  `GREEN REGENCY TEACUP AND SAUCER` and `ROSES REGENCY TEACUP AND SAUCER` are frequently purchased together (and also with the `PINK REGENCY` set).
    <!-- end list -->
      * **Action:** These items can be bundled, recommended as add-ons, or used in "complete the set" marketing emails.

## 6\. Challenges Faced and Solutions Implemented

  * **Challenge:** High percentage (24.93%) of missing `CustomerID`s.
      * **Solution:** Dropped these rows. While this reduces the dataset size, `CustomerID` is non-negotiable for user-based segmentation.
  * **Challenge:** Inconsistent product descriptions (e.g., " & " vs. "AND", "S/" vs. "SET OF", trailing punctuation).
      * **Solution:** A text-cleaning function was created to standardize all descriptions, ensuring that items were grouped correctly for market basket analysis.
  * **Challenge:** Inconsistent `StockCode` and `Description` mappings (e.g., one code with multiple descriptions).
      * **Solution:** Developed a function to resolve conflicts by selecting the most frequent (modal) description for each stock code (and vice-versa), creating a single source of truth.
  * **Challenge:** `StockCode`s with non-numeric values (e.g., 'POSTAGE', 'MANUAL') that represent charges, not products.
      * **Solution:** Filtered these non-product codes out of the data used for market basket analysis.
  * **Challenge:** Extreme outliers in `frequency` and `monetary` data skewed K-Means results.
      * **Solution:** Used `IsolationForest` to identify and remove the top 3% of outliers, resulting in more balanced and interpretable clusters.

## 7\. Future Improvements

  * **Feature Engineering Expansion:**
  * **Predictive Modeling:** Train predictive CLV (Customer Lifetime Value) models (e.g., BG/NBD, Gamma-Gamma) using the engineered RFMT features.
  * **Product Persona Clustering:** Use products purchased by customers to cluster them.
  * **Regional Analysis:** Incorporate the `Country` feature to perform segmentation and market basket analysis for different regions.
  * **Advanced Clustering:** Experiment with other clustering algorithms like HDBSCAN, which can handle noise and varying cluster densities without requiring a predefined K.
  * **NLP for Descriptions:** Use Natural Language Processing (NLP) techniques (e.g., TF-IDF, fuzzy matching) to programmatically group similar product descriptions instead of relying on manual regex rules.
  * **Deployment:** Create a simple interactive dashboard (e.g., using Tableau or Streamlit) to visualize the customer segments and cohort retention.
