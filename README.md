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

The analysis identified five distinct customer segments:

  * **Cluster 3: Best Customers (224 Customers)**
      * **Profile:** They have been customers the longest (high tenure, 347), have the longest active buying lifespan (high recency, 336), and buy the **most frequently** (14.4). They also have a **high Average Order Value (AOV)** (406). 
      * **Value:** They buy often *and* spend a lot when they do.
      * **Action:** Focus on loyalty rewards, exclusive access, and treating them as partners (e.g., soliciting feedback for new products).
       
  * **Cluster 2: Big-Ticket Spenders (250 Customers)**
      * **Profile:** They have **exceptionally high Average Order Value (AOV)** (911). They have a moderate tenure (268) and active lifespan (207). Their purchase frequency (3.0) is low, indicating they don't buy often. 
      * **Value:** They don't buy often, but when they do, they spend significantly more than any other group.
      * **Action:** Do not bombard them with low-value "buy now" messages. Target them with high-margin products, exclusive releases, and white-glove service. Their value is in *basket size*, not frequency.
        
  * **Cluster 1: Loyal Customers (809 Customers)**
      * **Profile:** They have a long tenure (338) and a long active lifespan (303). They purchase with good frequency (4.3) and have a moderate AOV (315). 
      * **Value:** They are the reliable repeat buyers.
      * **Action:** Nurture this relationship to maintain loyalty. Focus on community-building and personalized up-selling to increase their AOV and move them toward the VIP bracket.
       
  * **Cluster 0: At-Risk Customers (856 Customers)**
      * **Profile:** Despite a moderate tenure (261 days), their active lifespan (159) and frequency (2.1) are very low. They also have the **lowest AOV** (264). 
      * **Value:** They are the largest and least-engaged group. They made a couple of small repeat purchases but have likely disengaged.
      * **Action:** This group needs re-activation. Target them with compelling offers to increase their AOV or incentives to make their next purchase. Analyze *why* their engagement stopped.
   
  * **Cluster 4: New Repeat Customers (542 Customers)**
      * **Profile:** They have the **lowest tenure** (98), **shortest active lifespan** (60), and **lowest frequency** (1.8). Their AOV (269) is also very low.
      * **Value:** They just started their repeat-purchase journey. All of these metrics are low, which is expected for new customers.
      * **Action:** The goal is to convert them into long-term loyal customers (like Cluster 1). Focus on onboarding, demonstrating value, and encouraging a 3rd or 4th purchase.

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
