import FinanceDataReader as fdr
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로그 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# JSON 파일 저장 디렉토리 생성
json_dir = 'json_results'
os.makedirs(json_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_analysis.log'),
    level=logging.DEBUG,  # DEBUG 레벨로 모든 로그 기록
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔에도 로그 출력
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 콘솔에는 INFO 레벨 이상만 출력
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def fetch_stock_listing(market):
    """주식 종목 목록을 가져오는 함수."""
    try:
        logging.debug(f"{market} 종목 목록 가져오는 중...")
        return fdr.StockListing(market)['Code'].tolist()
    except Exception as e:
        logging.error(f"{market} 종목 목록 가져오기 중 오류 발생: {e}")
        return []

def fetch_stock_data(code, start_date, end_date):
    """주식 데이터를 가져오는 함수."""
    try:
        logging.debug(f"종목 코드 {code} 데이터 가져오는 중...")
        df = fdr.DataReader(code, start=start_date, end=end_date)
        logging.info(f"{code} 데이터 가져오기 성공")
        return code, df.to_dict(orient='records')
    except Exception as e:
        logging.error(f"{code} 데이터 가져오기 중 오류 발생: {e}")
        return code, None  # 오류 발생 시 None 반환

def fetch_and_save_stock_data(codes, start_date, end_date):
    """주식 데이터를 JSON 형식으로 가져와 저장하는 함수."""
    all_data = {}

    # 멀티스레딩으로 주식 데이터 가져오기
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_code = {executor.submit(fetch_stock_data, code, start_date, end_date): code for code in codes}
        for future in as_completed(future_to_code):
            code, data = future.result()
            if data is not None:
                all_data[code] = data

    # JSON 파일로 저장
    filename = os.path.join(json_dir, 'stock_data.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    logging.info(f"주식 데이터를 JSON 파일로 저장했습니다: {filename}")

def load_stock_data_from_json():
    """JSON 파일에서 주식 데이터를 로드하는 함수."""
    filename = os.path.join(json_dir, 'stock_data.json')
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_cup_with_handle(df):
    """컵과 핸들 패턴을 찾는 함수."""
    if len(df) < 60:  # 최소 60일의 데이터 필요
        logging.debug(f"데이터 길이가 60일 미만입니다. 종목 코드: {df['Code']}")
        return False, None
    
    cup_bottom = df['Low'].min()
    cup_bottom_index = df['Low'].idxmin()
    
    cup_top = df['Close'][:cup_bottom_index].max()
    
    handle = df.iloc[cup_bottom_index:cup_bottom_index + 10]  # 핸들 데이터 (10일)
    handle_top = handle['Close'].max()
    
    if handle_top < cup_top and cup_bottom < handle_top:
        logging.debug(f"패턴 발견! 종목 코드: {df['Code']}")
        return True, df.index[-1]  # 최근 날짜 반환
    return False, None

def search_cup_with_handle(stocks_data):
    """저장된 주식 데이터에서 Cup with Handle 패턴을 찾는 함수."""
    recent_cup_with_handle = None
    recent_date = None
    results = []

    for code, data in stocks_data.items():
        df = pd.DataFrame(data)
        df['Code'] = code  # 코드 추가

        is_pattern, pattern_date = is_cup_with_handle(df)
        if is_pattern:
            if recent_date is None or pattern_date > recent_date:
                recent_date = pattern_date
                recent_cup_with_handle = code
                results.append({
                    'code': code,
                    'pattern_date': pattern_date.strftime('%Y-%m-%d')
                })
    
    return recent_cup_with_handle, recent_date, results

# 메인 실행 블록
if __name__ == "__main__":
    logging.info("주식 분석 스크립트 실행 중...")
    
    today = datetime.today()
    start_date = today - timedelta(days=365)  # 최근 1년 전 날짜
    end_date = today.strftime('%Y-%m-%d')  # 오늘 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    # 멀티스레딩으로 종목 목록 가져오기
    markets = ['KOSPI', 'KOSDAQ']
    all_codes = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_market = {executor.submit(fetch_stock_listing, market): market for market in markets}
        for future in as_completed(future_to_market):
            market = future_to_market[future]
            try:
                codes = future.result()
                all_codes.extend(codes)
                logging.info(f"{market} 종목 목록 가져오기 성공: {len(codes)}개")
            except Exception as e:
                logging.error(f"{market} 종목 목록 가져오기 중 오류 발생: {e}")

    # 주식 데이터 가져와 JSON으로 저장
    fetch_and_save_stock_data(all_codes, start_date_str, end_date)

    # JSON 파일에서 주식 데이터 로드
    stocks_data = load_stock_data_from_json()

    # Cup with Handle 패턴 찾기
    recent_stock, date_found, results = search_cup_with_handle(stocks_data)
    if recent_stock:  # 최근 패턴이 발견된 경우
        logging.info(f"가장 최근 Cup with Handle 패턴이 발견된 종목: {recent_stock} (완성 날짜: {date_found})")
    else:
        logging.info("Cup with Handle 패턴을 가진 종목이 없습니다.")

    # 결과를 JSON 파일로 저장
    result_filename = os.path.join(json_dir, 'cup_with_handle_results.json')
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logging.info(f"결과를 JSON 파일로 저장했습니다: {result_filename}")
