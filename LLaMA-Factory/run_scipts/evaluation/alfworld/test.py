# import numpy as np
# import alfworld.agents.environment as environment
# import alfworld.agents.modules.generic as generic

# # load config
# config = generic.load_config()
# env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# # setup environment
# env = getattr(environment, env_type)(config, train_eval='train')
# env = env.init_env(batch_size=1)

# # interact
# obs, info = env.reset()
# while True:
#     # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
#     admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
#     random_actions = [np.random.choice(admissible_commands[0])]

#     # step
#     obs, scores, dones, infos = env.step(random_actions)
#     print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))

import os
# from openai import OpenAI
import openai

# 设置OpenAI API密钥和基础URL
# os.environ["OPENAI_API_KEY"] = "sk-aq1WIjHCe6JRibSVGboepeebCdvQTRFIu7WZDfENga3iuEzo"
# os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech"

# 创建OpenAI客户端
# client = OpenAI()
openai.api_key = "sk-aq1WIjHCe6JRibSVGboepeebCdvQTRFIu7WZDfENga3iuEzo"
openai.base_url = "https://api.chatanywhere.tech"

def call_gpt4o(prompt):
    try:
        # 调用GPT-4模型
        response = openai.chat.completions.create(
            model="gpt-4o-2024-08-06",  # 使用GPT-4模型
            messages=[
                {"role": "system", "content": "你是一个有帮助的AI助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        # 返回生成的回复
        return response.choices[0].message.content
    except Exception as e:
        print(f"调用GPT-4时出错: {e}")
        return None

# 测试函数
if __name__ == "__main__":
    # user_prompt = "请给我讲一个简短的笑话。"
    # response = call_gpt4o(user_prompt)
    # if response:
    #     print("GPT-4o的回复:", response)
    # else:
    #     print("无法获取回复。")
    import json
    json_path = '/mnt/tangyehui/dataset/mcts_datasets/alfworld_SFT_data/sft/correct_sft_data_single.json'
    with open(json_path,'r') as f:
        data = json.load(f)
    print(len(data))
