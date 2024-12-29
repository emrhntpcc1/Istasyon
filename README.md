# Passenger Count Prediction with LSTM

## 📜 Project Overview
This project aims to predict passenger count using Long Short-Term Memory (LSTM) networks. The model uses historical passenger data, time features (hour, day, etc.), and seasonal patterns (holiday days, weekdays) to forecast future passenger count. The solution combines data preprocessing, outlier handling, feature engineering, and time series forecasting.

## 🎯 Objectives
- **Outlier Detection**: Detect and clean outliers in the passenger data to ensure reliable predictions.
- **Feature Engineering**: Create new features, including year, month, day, and time-related features (such as sinusoidal transformations for cyclical features like hour) to enhance model performance.
- **Passenger Count Prediction**: Use LSTM to predict passenger count based on the preprocessed features.

## 🛠️ Key Features
- **Outlier Detection and Cleaning**: Outliers are detected using the Interquartile Range (IQR) method and cleaned to ensure the dataset is of high quality.
- **Feature Engineering**: Sinusoidal transformations for cyclical time data (hour, weekday) are applied to process time-related features more effectively.
- **LSTM Model**: A Long Short-Term Memory (LSTM) neural network is applied for time series predictions, with a Dropout layer used for regularization.
- **Feature Scaling**: Input features and target variables are scaled using MinMaxScaler for consistent scaling across the dataset.
- **Prediction Function**: A prediction function is provided to forecast passenger count based on user-defined inputs (time, holiday status, weekday).

## 🚀 Technologies Used
- **Programming Language**: Python  
- **Libraries**:
  - **Data Processing**: pandas, numpy  
  - **Data Visualization**: matplotlib, seaborn  
  - **Machine Learning**: TensorFlow (LSTM), scikit-learn  
  - **Data Scaling**: MinMaxScaler  
  - **Date Handling**: pandas  
- **AI Model**: LSTM (Long Short-Term Memory)

## 📊 Results
- **Data Preprocessing**: Outliers were cleaned, and a high-quality dataset was created.
- **LSTM Model Performance**: The model was trained on passenger data and achieved good prediction performance with low error.
- **Prediction Function**: Passenger count is successfully predicted based on various input features (time, holiday status, weekday).


🤝 Acknowledgements
We would like to thank our dear teacher Melih Ağraz for her support to the project. This work was completed as part of the "Yapay Zeka" course.

📬 Contact
Author: Emirhan Topçu , Emre Bektaş , Semih Cankat Cansız
Email: Emirhan: topcuemirhan59@gmail.com
       Emre: emrebkts2828@gmail.com
       Semih: cankat.cansiz@gmail.com
        
LinkedIn: Emirhan : https://www.linkedin.com/in/emirhan-top%C3%A7u-762825294/
          Emre : https://www.linkedin.com/in/emre-bekta%C5%9F-4414342a3/
          Semih : https://www.linkedin.com/in/semih-cankat-cans%C4%B1z-925821343/
Feel free to explore the repository and provide feedback!
