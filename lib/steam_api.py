import requests
import streamlit as st

@st.cache_data(ttl=3600)
def get_game_name(app_id: str) -> str:
    """Steam API를 사용하여 App ID에 해당하는 게임 이름을 가져옵니다."""
    if not app_id or not app_id.isdigit():
        return "알 수 없는 게임"
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&l=korean"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data[app_id].get('success'):
            return data[app_id]['data']['name']
        else:
            return f"게임 (ID: {app_id})"
    except Exception:
        return f"게임 (ID: {app_id})"