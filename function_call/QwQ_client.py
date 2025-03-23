import json
from openai import OpenAI
from functions import get_current_weather, get_n_day_weather_forecast, get_computing_cluster_status

functions_map = {
    "get_current_weather": get_current_weather,
    "get_n_day_weather_forecast": get_n_day_weather_forecast,
    "get_computing_cluster_status": get_computing_cluster_status,
}

# change the path to your specific models from HF or other sources
model = "/root/autodl-tmp/.cache/hub/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0/"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取当前城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称, e.g. 杭州市，深圳市",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "获取给定城市未来的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称, e.g. 杭州市，深圳市",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_computing_cluster_status",
            "description": "获取计算集群的状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "集群名称, e.g. AY88E，AY175M",
                    },
                },
                "required": ["cluster_name"],
            },
        },
    }
]

client = OpenAI(
    api_key="empty",
    base_url="http://0.0.0.0:8000/v1"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

num = 0
print("欢迎使用QwQ function_call demo!")

while True:
    num += 1
    print(f"第{num}轮对话: ")
    user_input = input("请输入您的问题: ")
    if user_input == "exit":
        print("感谢使用QwQ function_call demo!")
        break
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )
    print(f"----------Raw Response----------")
    print(response)
    print(f"----------Raw Response----------")
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        print("QwQ: ", response.choices[0].message.content)
        continue
    else:
        functon_name = tool_calls[0].function.name
        function_args = json.loads(tool_calls[0].function.arguments)
        result = functions_map[functon_name](**function_args)
        print(result)