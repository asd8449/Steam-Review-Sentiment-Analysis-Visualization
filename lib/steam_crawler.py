import requests
import pandas as pd
from datetime import datetime

def get_reviews_from_steam(app_id, num_reviews=1000):
    all_reviews = []
    cursor = '*'
    
    params = {
        'json': 1,
        'filter': 'all',
        'language': 'korean',
        'day_range': '365',
        'review_type': 'all',
        'purchase_type': 'all',
        'num_per_page' : 100,
    }

    for _ in range(num_reviews // 100):
        params['cursor'] = cursor
        
        try:
            response = requests.get(f"https://store.steampowered.com/appreviews/{app_id}", params=params)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            data = response.json()
            
            if data.get('success') != 1:
                print(f"API returned an error: {data}")
                break
                
            reviews = data.get('reviews', [])
            if not reviews:
                break
            
            for review in reviews:
                all_reviews.append({
                    'review': review['review'],
                    'voted_up': review['voted_up'],
                    'creation_time': datetime.fromtimestamp(review['timestamp_created']).strftime('%Y-%m-%d %H:%M:%S'),
                    'last_updated_time': datetime.fromtimestamp(review['timestamp_updated']).strftime('%Y-%m-%d %H:%M:%S'),
                })

            cursor = data.get('cursor')
            if not cursor:
                break

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    if not all_reviews:
        return None
        
    return pd.DataFrame(all_reviews)