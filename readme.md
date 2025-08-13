# Student Affairs Chatbot - AI Assistant

## Group members:
| Name | Student ID |
|------|------------|
| Christo Pananjickal Baby | 8989796 |
| Fasalu Rahman Kottaparambu | 8991782 |
| Srinu Babu Rai | 8994032 |

## Steps to run the project
1. Clone this(https://github.com/Christo-Conestoga-AIML/student-success-bot.git) repo to your local machine(you must have python 3.12 installed)
2. The trained models (vectors, logistic regression) is already present in models folder
3. To retrain the models, you can simply update the return of should_build_vector_db function in constants class
4. To connect with LLM you must create a .env file in root of project and add "OPENAI_API_KEY=your key" 
5. Create a virtual environment in python using python -m venv env
6. Install the requirements using pip install -r requirements.txt
7. Now you can run the app.py, which is the starting point of the application
8. You can then interact with the chatbot from the auto launched streamlit web app

## About the project
A presentation ppt file can be found at documents/Student Affairs Chatbot -AI Assistant.pptx 

### Problem statement
Students often face delays in getting answers about college policies, registration, fees, mental health, and campus life. Traditional support channels such as emails, calls, and office visits are slow, restricted to office hours, and can be confusing, while staff are overwhelmed by repetitive queries, leaving them with less time to handle serious or urgent issues.

### Proposed solution
We propose a smart, easy-to-use virtual assistant that provides instant answers 24/7, supports multiple languages, and escalates serious or sensitive issues to real human staff when necessary.
The data corpus is the data scraped from a college website.

## Implemented features
- Sentiment Detection & Escalation
- RAG for Interactions
- Next question prediction
- Multilingual Support

## Stake holders
- Students
- Student affairs staff
- College

## Techincal vendors
- Open AI
- Google translator
- ABC College Dataset

## Impact
- 24/7 instant support
- Reduces repetitive staff workload
- Multiple languages for inclusivity
- Quick follow-ups with next question suggestions

## NLP components
- **Normalization, Stopword filtering, Stemming** - Converting to lowercase, removing stopwords, words stemming
- **Tokenization** - Splitting data and prompt into individual words using NLTK
- **Vectorization** - Tokens are transformed into a numerical vector space from which correlated data can be found out using cosine similarity

## Prediction feature
- Predicts related questions at end of a conversation
- Uses Logistic Regression with re-ranking (sorting)
- Model trained once and saved
- Faster navigation

# Training and Evaluation Plots

This section shows the performance of the Logistic Regression model for predicting the next questions in the Student Affairs Chatbot. Each plot is explained with interpretation and examples.

---

### 1. Accuracy Curve
![Accuracy Curve](https://raw.githubusercontent.com/Christo-Conestoga-AIML/student-success-bot/refs/heads/master/plots/lr_accuracy_curve.png)

**What it shows:**  
This chart shows how well the Logistic Regression model performs on the training set vs the validation set over multiple epochs (iterations).

**Interpretation:**  
- Blue line: Training accuracy — how well the model predicts on the data it learned from.  
- Orange line: Validation accuracy — how well it predicts on unseen data.  
- Both lines are very high (~97%), indicating consistent accuracy.

**Example:**  
If a student asks "How do I pay my tuition fees?", the model correctly ranks the best next questions about 97 times out of 100 for both training and new data.

---

### 2. Precision–Recall Curve
![Precision–Recall Curve](https://raw.githubusercontent.com/Christo-Conestoga-AIML/student-success-bot/refs/heads/master/plots/lr_pr.png)

**What it shows:**  
Precision = of the questions predicted as “good suggestions,” how many were actually good.  
Recall = of all good suggestions in reality, how many the model successfully found.  
The curve shows the trade-off between these two.

**Key metric:**  
AP (Average Precision) = 0.912 → high precision while still capturing most relevant questions.

**Example:**  
If we suggest 10 “next questions” and 9 are actually useful, precision is high. Recall is high if most useful ones are included.

---

### 3. ROC Curve
![ROC Curve](https://raw.githubusercontent.com/Christo-Conestoga-AIML/student-success-bot/refs/heads/master/plots/lr_roc.png)

**What it shows:**  
Plots True Positive Rate (sensitivity) vs False Positive Rate (false alarms).  
The diagonal orange dashed line represents random guessing (AUC=0.5).  
Our curve is above it with AUC=0.980 → nearly perfect.

**Example:**  
The model can distinguish “good” next questions from irrelevant ones almost perfectly.

---

### 4. Confusion Matrix
![Confusion Matrix](https://raw.githubusercontent.com/Christo-Conestoga-AIML/student-success-bot/refs/heads/master/plots/lr_confusion.png)

**What it shows:**  
Compares predicted labels vs actual labels:  
- Top-left (661): Correctly predicted negatives (irrelevant questions identified correctly)  
- Top-right (10): False positives (predicted as relevant but weren’t)  
- Bottom-left (11): False negatives (missed relevant ones)  
- Bottom-right (63): Correctly predicted positives

**Example:**  
Out of all actual “good” suggestions, the model missed 11 and incorrectly suggested 10 bad ones.

---

### 5. Probability Histogram
![Probability Histogram](https://raw.githubusercontent.com/Christo-Conestoga-AIML/student-success-bot/refs/heads/master/plots/lr_prob_hist.png)

**What it shows:**  
Distribution of predicted probabilities for class 1 (relevant question).  
- Blue bars = actual positives → mostly near 1.0 (high confidence)  
- Orange bars = actual negatives → mostly near 0.0 (high confidence)

**Example:**  
If the model is 95% confident a question is relevant, it’s almost always correct.

---

### 6. Training Loss Curve
![Training Loss Curve](https://raw.githubusercontent.com/Christo-Conestoga-AIML/student-success-bot/refs/heads/master/plots/lr_train_curve.png)

**What it shows:**  
This chart shows how Log-Loss changes during training for both training and validation sets.

**Interpretation:**  
- Blue line: Training log-loss — error on the training data  
- Orange line: Validation log-loss — error on unseen data  
- Both lines stabilize at very low values (~0.095–0.097)  
- Small gap between lines → no overfitting

**Example:**  
If a student asks "How can I access funded training?", the model’s confidence in ranking correct next questions is high and stable.

---

### 7. Demo Ranking
![Demo Ranking](https://raw.githubusercontent.com/Christo-Conestoga-AIML/student-success-bot/refs/heads/master/plots/lr_demo_rank.png)

**What it shows:**  
An example of how the model ranks suggested next questions for a given student query.  
The most relevant suggestions appear at the top, demonstrating the practical effectiveness of the model.



## Prediction evaluation metrics
| Metric    | Value   |
|-----------|---------|
| Accuracy  | 97.18%  |
| Precision | 86.30%  |
| Recall    | 85.13%  |
| F1 Score  | 85.71%  |

## Software architecture
![Software Architecure](https://raw.githubusercontent.com/Christo-Conestoga-AIML/student-success-bot/refs/heads/master/images/SoftwareArchitecture.png)

## Future plans
- Add analytics dashboard - top rated questions, usage, prediction performance, response efficiency, find areas to improve
- Expand multilingual support
- Integrate with college student portal
