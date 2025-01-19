import os

from datetime import timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from icrawler.builtin import GoogleImageCrawler
from concurrent.futures import ThreadPoolExecutor
import time

def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def crawl_single_period(start_date, end_date, keywords, query, image_dir):
    
    """단일 날짜 범위에 대해 크롤링"""
    create_directory(f'{image_dir}/{keywords}/{start_date}')
    google_Crawler = GoogleImageCrawler(
        storage={'root_dir': f'{image_dir}/{keywords}/{start_date}'}, parser_threads=1, downloader_threads=1
    )
    google_Crawler.session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    filters = dict(date=(start_date, end_date))

    try:
        google_Crawler.crawl(
            keyword=keywords + " " + query,
            filters=filters,
            max_num=20
        )
    except TypeError as e:
        if 'NoneType' in str(e):
            print(f"TypeError 발생: {start_date} ~ {end_date} 기간의 이미지 skip")
            return
  

def crawl_image_per_item_concurrently(query, release_date, image_dir, keywords, num=None, max_workers=8):
    """이미지를 병렬로 크롤링하는 함수"""
    
    # 1년치 주별 날짜 리스트 생성
    end_date_list = [(release_date - timedelta(weeks=i)).date() for i in range(1, 53)]
    
    print(">"*20, keywords + " " + query)

    for i in tqdm(range(len(end_date_list)), desc="Downloading images sequentially..."):
        end_date = end_date_list[i]
        start_date = end_date - relativedelta(weeks=4)

        # 각 날짜 범위에 대해 크롤링
        crawl_single_period(start_date, end_date, keywords, query, image_dir)

        # 요청 간 대기 시간 추가 (Google의 차단을 방지)
        time.sleep(5)  # 10초 대기 (필요에 따라 조정 가능)