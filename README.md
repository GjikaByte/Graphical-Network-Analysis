# Graphical Network Analysis on Stock Returns Data

This repository contains the code, data, and results for a **graphical network analysis** of stock returns data. The study employs **advanced mathematical modeling** and **machine learning techniques** to explore the conditional dependencies and relationships between stock returns. By integrating **stochastic decision models** and **graphical lasso estimation**, this analysis investigates how stocks interact within a financial network.

---

## Dataset Overview

The analysis uses two datasets:
1. **Stock Returns Data**:
   - Source: CRSP via WRDS
   - Frequency: Daily, converted to Weekly
   - Date Range: 1999-12-31 to 2020-06-30

2. **3-Month T-Bill Data**:
   - Source: FRED
   - Frequency: Weekly
   - Date Range: max to 2020-06-26

Both datasets are cleaned and preprocessed before analysis. Preprocessed data is saved as `cleaned_data.xlsx`.

---

## Methodology

The core of this analysis is based on **mathematical modeling** using several advanced techniques:

### 1. Sparse Inverse Covariance Estimation
- **Model**: Graphical Lasso
- **Mathematical Focus**: Estimate the **precision matrix** (inverse of the covariance matrix), which reveals the conditional dependencies among the stock returns. This step highlights the significant pairwise interactions while suppressing noise.
- **Mathematical Model**:
  - **Covariance Matrix**: $\( \Sigma \)$ represents the variance-covariance structure.
  - **Precision Matrix**: $\( \Theta = \Sigma^{-1} \)$, representing conditional independence between assets.
  - **Optimization**: Solves the following objective:
$\hat{\Theta} = \underset{\Theta}{\text{argmin}} \left( \text{trace}(\Sigma \Theta) - \log \text{det}(\Theta) \right) + \lambda \| \Theta \|_1$



  - **Tool**: `GraphicalLassoCV` from `scikit-learn`, which imposes an L1 penalty to enforce sparsity in the precision matrix.

### 2. Clustering with Affinity Propagation
- **Model**: Affinity Propagation
- **Mathematical Focus**: Identify clusters of stocks that exhibit similar behaviors, based on pairwise similarity. The clustering is modeled using **responsibility** and **availability** matrices, which are iteratively updated.
- **Mathematical Model**: 
  - The **similarity matrix** $\( S \)$ quantifies the affinity between stocks $\( i \)$ and $\( j \)$, where a high value indicates a strong relationship. If similarity matrix is not given, apply Euclidean distance to calculate similarity matrix.
  - The responsibility $\( r(i,k) \)$ and availability $\( a(i,k) \)$ matrices are updated as:
    
    $r(i,k) = S(i,k) - \max_{k' \neq k} (a(i,k') + S(i,k'))$

    $a(i,k) = \min(0, r(k,k) + \sum_{i' \neq k} \max(0, r(i',k)))$
   - The sum of responsibility and availability matrix is criterion matrix which indicates the cluster label for each sample.

### 3. Manifold Learning and Dimensionality Reduction
- **Model**: Multi-Dimensional Scaling (MDS)
- **Mathematical Focus**: Reduce the high-dimensional relationships between stocks into two dimensions for visualization, preserving the distances (similarities) between stocks as accurately as possible in lower dimensions.
- **Mathematical Model**: Given a dissimilarity matrix $\( D \)$, find the configuration of points $\( X \)$ in 2D that minimizes:
  $\underset{X}{\text{minimize}} \sum_{i,j} (d_{ij} - \| X_i - X_j \|_2)^2$

---

## Results

### Summary of Key Findings:

- **Significant Conditional Dependencies**:  
  The **Graphical Lasso** method successfully identified direct correlations between certain stocks that were not apparent in the raw covariance matrix. This highlights hidden relationships that are critical for understanding stock interdependencies.

- **Sectoral Grouping**:  
  Clustering analysis revealed that stocks from similar sectors tend to exhibit more synchronized behavior. This supports the effectiveness of conditional dependency modeling in identifying sector-specific patterns and correlations among stocks within the same industry.

- **Network Structure**:  
  The **network graph** constructed from the precision matrix provided a clear visualization of stock interdependencies. It revealed how tightly related certain stocks are within their respective sectors, offering a deeper understanding of the financial network structure.

- **Dimensionality Reduction**:  
  The **Multi-Dimensional Scaling (MDS)** technique reduced the complexity of the high-dimensional relationships, offering an intuitive 2D visualization. This made it easier to identify and interpret stock clusters, providing insight into how stocks are grouped based on similar financial behaviors.

 ![Graphical Lasso](figures/Picture 1.png)

---
