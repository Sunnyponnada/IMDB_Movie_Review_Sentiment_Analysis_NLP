# IMDB_Movie_Review_Sentiment_Analysis_NLP
Sentiment analysis is a natural language processing (NLP) task that involves determining whether a given text expresses a positive or negative sentiment.

### Overview
 Sentiment analysis is a natural language processing (NLP) task that involves determining
 whether a given text expresses a positive or negative sentiment. In this project, we will
 analyze movie reviews from the IMDb dataset and predict the sentiment (positive or
 negative) based on the text of the reviews. By leveraging various text preprocessing
 techniques, feature extraction methods, and classification algorithms, this project will
 develop a machine learning model capable of accurately predicting the sentiment of movie
 reviews. The insights derived from this analysis can be useful for movie producers, critics,
 and platforms like IMDb to understand public opinion and tailor marketing or content
 strategies ac.cordingly

 ### Problem Statement
 The primary objective of this project is to build a machine learning classification model that
 can predict the sentiment of IMDb movie reviews. The dataset contains a collection of movie
 reviews, and each review is labeled as either positive or negative.
 Using text preprocessing, feature extraction techniques (such as TF-IDF), and various
 classification algorithms, the project will aim to develop a model that can effectively classify
 the sentiment of movie reviews. The model's performance will be evaluated using standard
 classification metrics, such as accuracy, precision, recall, and F.1-score

 ### Dataset Information
The IMDb dataset contains a large number of movie reviews, each labeled with either a  positive or negative sentiment
*  Text of the review: The actual review provided by the user.
*  Sentiment label: The sentiment of the review, either "positive" or "negative.".ve."

### Steps followed:
1. Data exploration and Preprocessing
2. Data cleaning
3. Text preprocessing
4. feature Engineering
5. Model Development
6. Model Evaluation
7. Prediction
![download](https://github.com/user-attachments/assets/6741c3aa-5e77-460d-b9ff-4e39fe0f6c4f)
![download](https://github.com/user-attachments/assets/0b6bfcc4-02b5-4d3b-81dd-73e185f362ec)

### Model Developments followed:
1. Logistic Regression
2. Naive Baye's
3. XGBoost
4. Random Forest
* ## Model-wise Confusion Matrix Insights
**Logistic Regression (Best Performing Model)**
* **High TP & TN:** Most reviews are correctly classified.
* **Low FP & FN:** Very few misclassified reviews.

 **Conclusion:** A well-balanced model with the least misclassifications and the best trade-off between precision and recall.

**Naive Bayes**
* **High TP:** Classifies positive reviews well.
* **Higher FP:** More false positives (overpredicts positivity).
 **Conclusion:** Performs well but struggles slightly with negative reviews, leading to more misclassifications.

**Random Forest**
* **Good TN:** Correctly predicts negative reviews.
* **Higher FN:** More false negatives (fails to detect positive sentiment).
 **Conclusion:** Slightly conservative in predicting positivity, leading to more Type II errors.

**XGBoost**
* **Balanced Performance:** Handles both classes well.
* **Moderate FN & FP:** Some misclassifications, but better than Random Forest.
 **Conclusion:** Works efficiently and is robust but does not outperform Logistic Regression significantly.*
![download](https://github.com/user-attachments/assets/4c1f8e91-cde0-450f-9e2c-7e8e1c4dfbb5)
![download](https://github.com/user-attachments/assets/3194fd57-fdc5-4c4e-a1f8-d9a4c23213fc)

### Overview
The ROC (Receiver Operating Characteristic) curve is a performance evaluation metric for classification models, plotting the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold levels. The **AUC (Area Under Curve)** score indicates how well a model distinguishes between positive and negative classes.

### Interpretation of the ROC Curve
From the given ROC curve:

**Logistic Regression (AUC = 0.96):**  
  - Achieves the highest AUC score, indicating it is the best-performing model in this analysis.  
  - Has a strong ability to separate positive and negative sentiments in IMDb reviews.
**Naive Bayes (AUC = 0.93):**  
  - Performs well but slightly underperforms compared to Logistic Regression.  
  - Works effectively with text classification due to its assumption of word independence.  

**Random Forest (AUC = 0.93):**  
  - Matches Naive Bayes in performance.  
  - Random Forest is robust, but it may not be the best for high-dimensional text data.  

**XGBoost (AUC = 0.93):**  
  - Similar to Random Forest and Naive Bayes in AUC score.  
  - Known for handling structured data well but does not significantly outperform simpler models in this case.

### Conclusion
**Logistic Regression outperforms all other models with an AUC of 0.96**, making it the best choice for this sentiment classification task.  
**Tree-based models (Random Forest, XGBoost) and Naive Bayes perform similarly (AUC = 0.93)**, indicating they are also viable alternatives.  
**AUC values above 0.90 suggest all models perform well** in distinguishing between positive and negative IMDb reviews.
