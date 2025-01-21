import asyncio
import os
import re
import argparse
import json
from threading import Thread
from typing import Any, Dict, List, Optional
from copy import deepcopy
import sys
import datetime
import random
import openai
import numpy as np
import torch
            

import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
from src.llamafactory.chat.base_engine import BaseEngine, Response
from src.llamafactory.chat.hf_engine import HuggingfaceEngine
from src.llamafactory.chat.vllm_engine import VllmEngine
from src.llamafactory.hparams import get_infer_args

MAX_ROUND = 40
# TODO: 参考DPO推理脚本，改模型加载部分代码
def parse_args():
    parser = argparse.ArgumentParser(description="Run ALFWorld with ChatModel")
    parser.add_argument("--config_file", type=str, default="evaluation/alfworld/base_config.yaml", help="Path to the config file")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--adapter_name_or_path", type=str, help="Path to the adapter (LoRA) weights")
    parser.add_argument("--finetuning_type", type=str, default="lora", help="Finetuning type (e.g., 'lora')")
    parser.add_argument("--template", type=str, default="llama3", help="Template for the model")
    parser.add_argument("--save_path", type=str, default="results.json", help="Path to save the results")
    return parser.parse_args()

def load_alf_config(config_file: str):
    original_argv = sys.argv
    sys.argv = [sys.argv[0], config_file]
    
    try:
        config = generic.load_config()
    finally:
        sys.argv = original_argv
    
    return config

def convert_messages_mcts(messages: List[Dict[str, str]]):
    # MCTS版本的messages
    action_idx = 0
    if len(messages) == 1:
        user_content = f"Init environment: {messages[0]['observation']}\n The candidate actions are {messages[0]['action_space']}"
    else:
        user_content = f"Init environment: {messages[0]['observation']}\n"
        for message in messages[1:]:
            user_content += f"Action {action_idx} is {message['action']}. Action {action_idx}'s  observation: {message['observation']} "
            action_idx += 1
        
        # 添加action_space
        user_content += f"Decide the next action, The candidate actions are {message['action_space']}.Your response must be exactly one action name chosen strictly from provide candidate actions.Do not provide any explanations. "
    return [{"role": "user", "content": user_content}]




class ALFWorldChatModel:
    def __init__(self, engine: BaseEngine, config: Dict[str, Any], seed: int = 42):
        self.engine = engine
        env_type = config['env']['type']
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.train_eval = 'eval_out_of_distribution'
        self.env = getattr(environment, env_type)(config, train_eval=self.train_eval)
        # self.env = getattr(environment, env_type)(config, train_eval='train') # train/eval_in_distribution/eval_out_of_distribution
        self.env = self.env.init_env(batch_size=1)
        print(f"Environment type: {env_type}")
        # print(f"Number of tasks in json_file_list: {len(self.env.json_file_list)}")
        print(f"Batch size: {self.env.batch_size}")
        self.reset_environment()


    def reset_environment(self):
        self.obs, self.info = self.env.reset()
        return self.obs, self.info
    
    def convert_messages(self, messages: List[Dict[str, str]], delete_all_candidate_actions: bool = True):
        for message in messages:
            # delete the candidate actions in the middle of the conversation
            if not delete_all_candidate_actions:
                if message["role"] == "user" and message != messages[-1]:
                    new_content = "\n".join([item.strip() for item in message["content"].split("\n") if not item.startswith("The candidate")])
                    message["content"] = new_content
            # delete the candidate action in all messages
            else:
                if message["role"] == "user" and delete_all_candidate_actions:
                    new_content = "\n".join([item.strip() for item in message["content"].split("\n") if not item.startswith("The candidate")])
                    message["content"] = new_content
        return messages

    async def run_step(self, messages: List[Dict[str, str]], system) -> Dict[str, Any]:
        try:
            messages = self.convert_messages(messages)
            print(f"messages:{messages}")
            count = 0 
            while count <=10:
                # messages = convert_messages_mcts(messages)
                responses = await self.engine.chat(messages, system)
                action = responses[-1].response_text.strip().strip("'")
                
                pattern = r'\baction\s+\d+\s+is\s+' # 去掉action num is
                action = re.sub(pattern, '', action)
                pattern = r'\bAction\s+\d+\s+is\s+' # 去掉Action num is
                action = re.sub(pattern, '', action)
                pattern = r'^action\s+\d+:\s+'
                action = re.sub(pattern, '', action)
                pattern = r'^Action\s+\d+:\s+'
                action = re.sub(pattern, '', action)
                pattern = r'\baction\s+'
                action = re.sub(pattern, '', action)

                if "action:" in action or ">" in action:
                    action = action.replace("action:", "").replace(">", "").strip()

                if 'spatul1' in action:
                    action = action.replace("spatul1",'spatula 1')# alfworld环境错误，给的candidate action不对
                count += 1

                if 'assistant' not in action and 'user' not in action:
                    break

            new_obs, scores, dones, infos = self.env.step([action])

            if "Nothing happens" in new_obs or 'close cabinet' in action:
                if action in messages[-1]['content']:
                    messages[-1]['content'].replace(action,'')
                
                # 再来一次
                responses = await self.engine.chat(messages, system)
                action = responses[-1].response_text.strip().strip("'")
                print(f"new_action:{action}")

                new_obs, scores, dones, infos = self.env.step([action])

            self.obs = new_obs
            self.info = infos


            return {
                "action": action,
                "observation": self.obs[0],
                "score": infos['goal_condition_success_rate'] if 'goal_condition_success_rate' in infos else [0.0],
                "done": dones[0],
                "info": self.info
            }
        except Exception as e:
            print(f"Error during step execution: {e}")
            return None

def _start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()



class GPT4Engine(BaseEngine):
    def __init__(self, model_args, data_args, finetuning_args, generating_args):
        super().__init__(model_args, data_args, finetuning_args, generating_args)
        openai.api_key = "sk-aq1WIjHCe6JRibSVGboepeebCdvQTRFIu7WZDfENga3iuEzo"
        openai.base_url = "https://api.chatanywhere.tech"
        self.model = "gpt-4o-2024-08-06"


    async def chat(self, messages: List[Dict[str, str]], system: str = "") -> List[Dict[str, Any]]:
        try:
            if system:
                messages = [{"role": "system", "content": system}] + messages
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages]
            )

            return [Response(
                response_text=response.choices[0].message.content,
                response_length=len(response.choices[0].message.content),
                prompt_length=sum(len(m["content"]) for m in messages),
                finish_reason="stop"
            )]
        except Exception as e:
            print(f"Error in GPT4Engine chat: {e}")
            return []
        
    async def stream_chat(self, messages: List[Dict[str, str]], system: str = ""):
        raise NotImplementedError("stream_chat is not implemented for GPT4Engine")

    def get_scores(self, texts: List[str], **kwargs) -> Dict[str, List[float]]:
        raise NotImplementedError("get_scores is not implemented for GPT4Engine")

class ChatModel:
    def __init__(self, args: Dict[str, Any]) -> None:
        hf_args = deepcopy(args)
        
        config_file = hf_args.pop('config_file', None)
        save_path = hf_args.pop('save_path', None)
                
        model_args, data_args, finetuning_args, generating_args = get_infer_args(hf_args)
        if hf_args['template'] == "gpt4":
            self.engine: BaseEngine = GPT4Engine(model_args, data_args, finetuning_args, generating_args)
        else:
            if model_args.infer_backend == "huggingface":
                self.engine: BaseEngine = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
                print("model layer6 q_proj weight is")
                print(self.engine.model.model.layers[6].self_attn.q_proj.weight)  # 打印权重
            elif model_args.infer_backend == "vllm":
                self.engine: BaseEngine = VllmEngine(model_args, data_args, finetuning_args, generating_args)
            else:
                raise NotImplementedError(f"Unknown backend: {model_args.infer_backend}")

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()        
        
        config = load_alf_config(config_file)
        self.alfworld_model = ALFWorldChatModel(self.engine, config)
        
        self.save_path = save_path

    async def run_alfworld_step(self, messages: List[Dict[str, str]], system,) -> Dict[str, Any]:
        return await self.alfworld_model.run_step(messages, system)

    def reset_alfworld(self) -> str:
        return self.alfworld_model.reset_environment()

async def run_chat(args: argparse.Namespace) -> None:
    chat_model = ChatModel(vars(args))
    print("Welcome to the ALFWorld CLI. Running tasks automatically.")
    results = []
    all_tasks_info = []
    task_count = 0

    system_message = f"""
    You are an AI robot agent in an interactive environment. Your goal is to accomplish the given task through a series of actions. Follow these guidelines:
    1. Carefully analyze the task requirements. Break down complex tasks into smaller, manageable steps and create a mental plan before acting.
    2. Be persistent in searching for required objects.When searching for objects, use common sense to predict likely locations, then systematically explore those areas.
    3. For tasks involving multiple objects, keep a mental count of how many you've collected or placed. For tasks requiring multiple identical items (such as "put two items"), ensure that you actually find and place two different items, rather than repeatedly using the same item.
    4. Avoid repeating the same action consecutively. If an action doesn't work, explore other objects or locations instead of retrying the same action.
    5. Your response must be exactly one action name chosen strictly from provide candidate actions.Do not provide any explanations.
    """

    while True:
        round = 0
        task_count += 1
        obs, info = chat_model.reset_alfworld()
        game_file = info['extra.gamefile'][0]
        current_obs = obs[0]
        messages = []
        
        init_action_space = [action for action in info["admissible_commands"][0] if not action.startswith("look")]
        done = False
        task_history = []

        messages.append({"role": "user", "content": f"Init environment: {current_obs}\n The candidate actions are {init_action_space}"})

        while not done:
            round += 1

            step_result = await chat_model.run_alfworld_step(messages, system_message)

            if step_result is None:
                print("Error occurred during step execution. Resetting environment.")
                break


            print(f"Action {round}: {step_result['action']}")
            print(f"Observation {round}: {step_result['observation']}")
            
            task_history.append({
                "action": step_result['action'],
                "observation": step_result['observation'],
                "score": step_result['score'],
                "done": True if step_result['done'] else False
            })

            current_action_space = [action for action in step_result['info']["admissible_commands"][0] if not action.startswith("look")]

            messages.append({"role": "assistant", "content": step_result['action']})

            messages.append({"role": "user", "content": f"Current observation: {step_result['observation']}\n The candidate actions are {current_action_space}"})

            current_obs = step_result['observation']
            done = step_result['done']
            if round >= MAX_ROUND:
                break


        exceeded_max_round = round >= MAX_ROUND
        all_tasks_info.append({
            "task_id": game_file,
            "messages": messages,
            "history": task_history,
            "final_score": step_result['score'][0] if step_result else 0,
            "completed": True if done else False,
            "exceeded_max_round": exceeded_max_round
        })

        if done:
            print(f"\nTask {task_count} completed \n")
        else:
            print(f"\nTask {task_count} failed \n")

        results.append({
            "task_id": game_file,
            "final_score": step_result['score'][0] if step_result else 0,
            "completed": True if done else False,
            "exceeded_max_round": exceeded_max_round
        })

        if task_count >= 60 or task_count>=chat_model.alfworld_model.env.num_games: #chat_model.alfworld_model.env.num_games:
            break

    success_rate = sum(1 for r in results if r['completed']) / len(results)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all success tasks' information
    success_tasks_file = f"tasks_success_{chat_model.alfworld_model.train_eval}_{args.template}_{current_time}.json"
    success_tasks_save_path = os.path.join(chat_model.save_path, success_tasks_file)
    os.makedirs(os.path.dirname(success_tasks_save_path), exist_ok=True)
    success_task_info = [item for item in all_tasks_info if item['completed']]
    with open(success_tasks_save_path, 'w') as f:
        json.dump(success_task_info, f, indent=4)
    print(f"Success tasks' information saved to {success_tasks_save_path}")

    # Save all failed task's information
    fail_tasks_file = f"tasks_fail_{chat_model.alfworld_model.train_eval}_{args.template}_{current_time}.json"
    fail_tasks_save_path = os.path.join(chat_model.save_path, fail_tasks_file)
    os.makedirs(os.path.dirname(fail_tasks_save_path), exist_ok=True)
    fail_task_info = [item for item in all_tasks_info if not item['completed']]
    with open(fail_tasks_save_path, 'w') as f:
        json.dump(fail_task_info, f, indent=4)
    print(f"Fail tasks' information saved to {fail_tasks_save_path}")

    # Save overall results
    overall_file = f"overall_results_{chat_model.alfworld_model.train_eval}_{args.template}_{current_time}.json"
    overall_save_path = os.path.join(chat_model.save_path, overall_file)
    with open(overall_save_path, 'w') as f:
        json.dump({
            "success_rate": success_rate,
            "results": results,
        }, f, indent=4)


    print(f"Overall task results saved to {overall_save_path}")
    print(f"Final success rate: {success_rate:.2%}")

    success_length = []
    for item in success_task_info:
        success_length.append(len(item['history']))
    print(f"Average completed steps: {np.mean(success_length)}")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_chat(args))
