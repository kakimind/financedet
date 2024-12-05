import pandas as pd
import joblib
import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_data_fetcher.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    logging.debug("주식 데이터를 가져오는 중...")
    try:
        file_path = os.path.join('data', 'stock_data_with_indicators.csv')
        logging.debug(f"CSV 파일 경로: {file_path}")

        dtype = {
            'Code': 'object',
            'Date': 'str',
            'Open': 'float',
            'High': 'float',
            'Low': 'float',
            'Close': 'float',
            'Volume': 'float',
            'Change': 'float',
            'MA5': 'float',
            'MA20': 'float',
            'MACD': 'float',
            'MACD_Signal': 'float',
            'Bollinger_High': 'float',
            'Bollinger_Low': 'float',
            'Stoch': 'float',
            'RSI': 'float',
            'ATR': 'float',
            'CCI': 'float',
            'EMA20': 'float',
            'EMA50': 'float',
            'Momentum': 'float',
            'Williams %R': 'float',
            'ADX': 'float',
            'Volume_MA20': 'float',
            'ROC': 'float',
            'CMF': 'float',
            'OBV': 'float',
        }

        df = pd.read_csv(file_path, dtype=dtype)
        logging.info(f"주식 데이터를 '{file_path}'에서 성공적으로 가져왔습니다.")
        logging.debug(f"데이터프레임 정보:\n{df.info()}")

        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        logging.debug(f"가져온 데이터의 첫 5행:\n{df.head()}")
        return df
    except Exception as e:
        logging.error(f"주식 데이터 가져오기 중 오류 발생: {e}")
        return None

def prepare_data(df):
    """데이터를 준비하고 분할하는 함수."""
    logging.debug("데이터 준비 및 분할 시작...")
    
    features = [
        'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
        'EMA20', 'EMA50', 'ATR', 'Volume'
    ]

    X = []
    y = []
    stock_codes = []

    # 각 종목에 대해 6개월치 데이터를 가지고 훈련
    for stock_code in df['Code'].unique():
        stock_data = df[df['Code'] == stock_code]
        stock_data = stock_data.sort_values(by='Date')  # 날짜 기준으로 정렬

        # 6개월 데이터 필터링
        recent_data = stock_data[stock_data['Date'] >= (pd.Timestamp.now() - pd.DateOffset(months=6))]

        # 마지막 날의 피처와 타겟을 함께 추가
        for i in range(len(recent_data) - 26):  # 26일 예측
            target_today = 1 if recent_data['Close'].iloc[i + 26] > recent_data['Close'].iloc[i] * 1.29 else 0  # 29% 상승 여부
            
            # 피처 추가
            X.append(recent_data[features].iloc[i].values)  # 현재 피처
            y.append(target_today)  # 오늘의 타겟 값
            stock_codes.append(stock_code)  # 종목 코드 추가

    X = np.array(X)
    y = np.array(y)

    # 클래스 분포 확인
    logging.info(f"타겟 클래스 분포: {np.bincount(y)}")

    # SMOTE 적용
    if len(np.unique(y)) > 1:
        logging.debug("SMOTE를 적용하여 클래스 불균형 문제 해결 중...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info("SMOTE를 적용하여 데이터 샘플 수를 조정했습니다.")
    else:
        logging.warning("타겟 클래스가 1개만 존재합니다. SMOTE를 적용하지 않습니다.")
        X_resampled, y_resampled = X, y

    # 데이터 분할
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )
    
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    logging.debug("데이터 준비 및 분할 완료.")
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train_model_with_hyperparameter_tuning():
    """모델을 훈련시키고 하이퍼파라미터를 튜닝하는 함수."""
    logging.info("모델 훈련 시작...")
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 모델 훈련을 중단합니다.")
        return None

    # 데이터 준비 및 분할
    X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_data(df)

    # 하이퍼파라미터 튜닝을 위한 GridSearchCV 설정
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
    
    logging.info("하이퍼파라미터 튜닝 시작...")
    grid_search.fit(X_train, y_train)  # 하이퍼파라미터 튜닝
    
    # 최적의 하이퍼파라미터 출력
    logging.info(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
    print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")

    # 최적의 모델로 재훈련
    best_model = grid_search.best_estimator_

    # 모델 평가
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info(f"모델 성능 보고서:\n{report}")
    print(report)

    logging.info("모델 훈련 완료.")
    return best_model  # 최적 모델 반환

def predict_future_days(model):
    """향후 26일의 상승 여부를 예측하는 함수."""
    logging.info("향후 26일 예측 시작...")
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 예측을 중단합니다.")
        return None
        
    # features 리스트 정의
    features = [
        'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
        'EMA20', 'EMA50', 'ATR', 'Volume'
    ]

    # 오늘 데이터 가져오기
    today_data = df.tail(1)
    if today_data.empty:
        logging.warning("오늘 데이터가 없습니다. 예측을 중단합니다.")
        return

    future_predictions = []
    
    # 향후 26일 동안 예측
    for i in range(1, 27):  # 1일부터 26일까지 예측
        future_data = today_data.copy()
        future_data['Date'] += pd.DateOffset(days=i)  # 날짜를 1일씩 더함
        
        # 예측을 위한 피처 데이터 준비
        future_features = future_data[features].values
        
        prediction = model.predict(future_features)
        future_predictions.append((future_data['Date'].values[0], prediction[0]))  # 날짜와 예측값 저장

    logging.info("향후 26일 예측 완료.")
    return future_predictions  # 날짜와 예측값 리스트 반환

# 모델 훈련 및 예측 실행
best_model = train_model_with_hyperparameter_tuning()
if best_model:
    future_predictions = predict_future_days(best_model)
    if future_predictions is not None:
        for date, prediction in future_predictions:
            logging.info(f"{date.date()}: 예측된 상승 여부: {'상승' if prediction == 1 else '하락'}")

