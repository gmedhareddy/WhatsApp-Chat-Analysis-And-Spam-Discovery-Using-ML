# WhatsApp-Chat-Analysis-And-Spam-Discovery-Using-ML
📊 WhatsApp Chat Analysis and Spam Discovery Using Machine Learning

This project presents an end-to-end machine learning system to analyze WhatsApp chat data and automatically detect spam messages using Natural Language Processing (NLP) and Machine Learning (ML) techniques.
In addition to spam detection, the system provides insightful visual analytics to understand user behavior, message patterns, and chat activity.

🎯 Project Objectives

Analyze exported WhatsApp chat data (.txt format)

Detect and classify messages as Spam or Ham (Non-Spam)

Apply NLP preprocessing to handle informal chat language

Build and evaluate ML classifiers (Naïve Bayes, SVM)

Provide EDA visualizations such as pie charts, heatmaps, bar charts, and word clouds

Develop a user-friendly web interface for analysis

🧠 Technologies Used
Programming & Libraries

Python

Pandas – Data manipulation

NumPy – Numerical operations

Scikit-learn – ML algorithms & evaluation

NLTK / TextBlob – Text preprocessing & sentiment analysis

Matplotlib & Seaborn – Data visualization

WordCloud – Text visualization

Web Framework

Streamlit (or Flask) – Interactive web interface

🏗️ System Architecture

Workflow:

Upload WhatsApp chat file (.txt)

Parse chat into structured format

Clean & preprocess text data

Feature extraction using TF-IDF

Train ML classifiers

Spam prediction (Spam / Non-Spam)

Visualization & reporting

🗂️ Project Structure
WhatsApp-Chat-Analysis-ML/
│
├── app.py                  # Streamlit / Flask web app
├── data_parser.py          # Chat parsing & preprocessing
├── spam_detector.py        # ML model training & prediction
├── eda_visualizer.py       # EDA & visualization functions
│
├── model/
│   ├── spam_model.pkl      # Trained ML model
│   └── tfidf_vectorizer.pkl
│
├── dataset/
│   ├── spam.csv            # Kaggle SMS spam dataset
│
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

📁 Dataset Used

Kaggle SMS Spam Collection Dataset

Labels:

spam – Promotional / malicious messages

ham – Legitimate messages

WhatsApp exported chat files (.txt) are analyzed using the trained model

The Kaggle dataset is used to train the classifier, which is then applied to WhatsApp chats. 

dp1

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/WhatsApp-Chat-Analysis-ML.git
cd WhatsApp-Chat-Analysis-ML

2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ How to Run the Project
Run Web Application
streamlit run app.py


or (if Flask is used)

python app.py


Then open the browser and upload:

WhatsApp chat export (.txt)

📊 Features & Outputs
🔍 Spam Detection

Classifies messages into:

Spam

Non-Spam

Uses TF-IDF + Naïve Bayes / SVM

Displays accuracy, precision, recall, and F1-score

📈 Exploratory Data Analysis (EDA)

Message distribution by user

Spam vs Ham pie chart

Heatmap of activity (hour/day)

Most frequent words

Word clouds for spam messages

😊 Sentiment Analysis (Optional)

Positive / Neutral / Negative message classification

📌 Results

Naïve Bayes achieved high accuracy for short text classification

ML models significantly outperform rule-based spam filters

Effective handling of informal language, links, and spam patterns

Visual analytics provide actionable insights into chat behavior

🔐 Privacy & Ethics

Works only on user-uploaded chat files

No data is stored or shared externally

Designed with data anonymization and privacy considerations

Suitable for academic and personal use

🚀 Future Enhancements

Real-time spam detection

Deep learning models (LSTM / BERT)

Multilingual chat analysis

Cloud deployment (AWS / Heroku)

WhatsApp group-level spam behavior analysis
