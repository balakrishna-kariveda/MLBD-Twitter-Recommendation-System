# Twitter Recommendation System 

**Author**: Sachchida Nand Tiwari  
**Roll No**: M23CSA527  
**Date**: September 2024  

A simplified implementation of Twitter's recommendation system focusing on:  
- Community detection through clustering  
- Engagement prediction using ML  
- Content diversification rules  

## Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Results](#-results)
- [Future Work](#-future-work)

## Features
1. **Synthetic Data Generation**  
    - 1K users with verification status  
    - 5K tweets across 5 categories  
    - 20K user interactions (views/likes/retweets)  

2. **Clustering Engine**  
    ```python
    kmeans = KMeans(n_clusters=5)
    user_topics['cluster'] = kmeans.fit_predict(user_topics)
    ```

3. **Candidate Sourcing**  
    - Cluster-based recommendations  
    - Popular tweets  
    - Verified content  

4. **Ranking Model**  
    ```python
    model = Pipeline([
         ('preprocessor', ColumnTransformer(...)),
         ('classifier', RandomForestClassifier())
    ])
    ```

## Installation
Clone the repository:  
```bash
git clone https://github.com/sntiwari1/MLBD.git
cd MLBD
```

Install dependencies:  
```bash
pip install numpy pandas faker scikit-learn nltk
```

Download NLTK resources:  
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage
Generate synthetic data:  
```python
users, tweets, interactions = generate_synthetic_data()
```

Run the full recommendation pipeline:  
```python
target_user = np.random.choice(users['author_id'])
recommendations = recommend_tweets(target_user, users, tweets, interactions, model)
```

View recommendations:  
```python
print(recommendations[['tweet_id', 'content', 'predicted_score']].head(10))
```

## Architecture
- **Diagram**:  
[Architecture Diagram](./architecture.png)

## Results
**Output**  
Recommendations for user 134:

| tweet_id | content                                                                 | predicted_score |
|----------|-------------------------------------------------------------------------|-----------------|
| 3953     | Why build degree. Capital beautiful factor mot...                       | 0.972337        |
| 1188     | Back but develop position. Focus buy Republica...                       | 0.951104        |
| 2977     | Standard federal time market half heavy option...                       | 0.933842        |
| 3999     | Effect effort during position. Until street re...                       | 0.902509        |
| 1733     | Development him expect former begin past. Comp...                       | 0.892429        |
| 1128     | Page friend theory team hotel article surface....                       | 0.886283        |
| 2626     | Score song leave face memory. Out soon total b...                       | 0.878132        |
| 973      | If leader final book finish race. Mention oper...                       | 0.877452        |
| 4106     | Method many toward your. Do little item. Reaso...                       | 0.864778        |
| 365      | Crime radio call study cut. Kind all some char...                       | 0.862481        |
