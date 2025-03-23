import requests

def get_current_weather(city):
    # api_key = "YOUR_API_KEY"
    # base_url = "BASE_URL"
    # params = {
    #     'key': api_key,
    #     'city': city,
    #     'extensions': 'base', # 实时天气
    #     'output': 'json'
    # }
    # response = requests.get(base_url, params=params)
    # data = response.json()
    weather = {
        '城市': f'{city}',
        '温度': '25℃',
        '天气状况': '多云',
        '风向': '东北风',
        '风力': '3级',
        '湿度': '60%',
        '报告时间': '2023-08-08 12:00:00'
    }
    return weather

def get_n_day_weather_forecast(city):
    forecasts = []
    for i in range(1, 10):
        forecast = {
            '城市': f'{city}',
            '温度': '25℃',
            '天气状况': '多云',
            '风向': '东北风',
            '风力': '3级',
            '湿度': '60%',
            '报告时间': f'2023-08-0{i} 12:00:00'
        }
        forecasts.append(forecast)
    return forecasts

def get_computing_cluster_status(cluster_name):
    status = {
        '集群名称': f'{cluster_name}',
        '节点数量': '10',
        '任务数量': '100',
        'CPU使用率': '60%',
        '内存使用率': '80%',
        '磁盘使用率': '90%',
        '报告时间': '2023-08-08 12:00:00'
    }
    return status