import asyncio
import os
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
            responses = await self.engine.chat(messages, system)
            action = responses[-1].response_text.strip().strip("'")
            if "action:" in action or ">" in action:
                action = action.replace("action:", "").replace(">", "").strip()

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

def get_few_shot_data(game_name):
    with open('evaluation/alfworld/few_shot.json', 'r') as f:
        few_shot_data = json.load(f)
    task_prefixes_map = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'
    }
    few_shot_name = ""
    for task_name, task_value in task_prefixes_map.items():
        if task_name in game_name:
            few_shot_name = task_value
    few_shot = few_shot_data[f"act_{few_shot_name}_0"]
    return few_shot


async def run_chat(args: argparse.Namespace) -> None:
    chat_model = ChatModel(vars(args))
    print("Welcome to the ALFWorld CLI. Running tasks automatically.")
    results = []
    all_tasks_info = []
    task_count = 0

    while True:
        round = 0
        task_count += 1
        obs, info = chat_model.reset_alfworld()
        game_name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        current_obs = obs[0]
        few_shot = get_few_shot_data(game_name)
        messages = []
        
        # with few shot
        # system_message = f"""
        # You are an AI robot agent in an interactive environment, tasked with completing specific objectives. Your goal is to accomplish the given task through a series of precise and effective actions. Follow these guidelines:
        # 1. Carefully analyze the task requirements. Break down complex tasks into smaller, manageable steps and create a mental plan before acting.
        # 2. Your response should consist solely of a single action chosen from the list of candidate actions. Do not provide explanations.
        # 3. Prioritize efficiency: minimize unnecessary movements and actions. If you've already opened a container that you need to use again, remember its location and state.
        # 4. Pay close attention to environmental feedback. If an action is ineffective, do not repeat it; instead, seek alternative approaches or explore other locations, and do not explore where you have already searched.
        # 5. When encountering obstacles, attempt innovative solutions, such as using different items or exploring new locations. Be persistent in searching for required objects.
        # 6. When searching for objects, use common sense to predict likely locations, then systematically explore those areas.
        # 7. For tasks involving multiple objects, keep a mental count of how many you've collected or placed.
        # 9. If you need to pick up an object, first go to the container that holds it. Remember, you can only hold one item at a time.
        # 10. For heating or cooling items, go directly to the appliance and use the appropriate action. Do not put the object in the appliance or open it unnecessarily.
        # 11. If you find yourself stuck or in a loop, reassess your strategy and consider unexplored areas or actions.
        # 12. For tasks requiring multiple identical items (such as "put two items"), ensure that you actually find and place two different items, rather than repeatedly using the same item.
        # 13. Do not repeat the same action.
        # There is a example: {few_shot}
        # """
        
        # without few shot
        system_message = f"""
        You are an AI robot agent in an interactive environment. Your goal is to accomplish the given task through a series of actions. Follow these guidelines:
        1. Carefully analyze the task requirements. Break down complex tasks into smaller, manageable steps and create a mental plan before acting.
        2. Be persistent in searching for required objects.When searching for objects, use common sense to predict likely locations, then systematically explore those areas.
        3. For tasks involving multiple objects, keep a mental count of how many you've collected or placed. For tasks requiring multiple identical items (such as "put two items"), ensure that you actually find and place two different items, rather than repeatedly using the same item.
        4. Avoid repeating the same action consecutively. If an action doesn't work, explore other objects or locations instead of retrying the same action.
        5. Your response must be exactly one action name chosen strictly from provide candidate actions.Do not provide any explanations.
        """

        init_action_space = [action for action in info["admissible_commands"][0] if not action.startswith("look")]
        done = False
        task_history = []
        # print(f"\nNew task started. Initial observation: {obs[0]}")
        messages.append({"role": "user", "content": f"Init environment: {current_obs}\n The candidate actions are {init_action_space}"})
        # messages.append({"role": "user", "content": f"Current observation: {current_obs}"})
        # messages.append({"role": "user", "content": f"你好"})


        print(f"system_message: {system_message}")
        print(f"messages: {messages}")
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
            # print(f"messages: {messages[-1]}")
            messages.append({"role": "user", "content": f"Current observation: {step_result['observation']}\n The candidate actions are {current_action_space}"})
            # messages.append({"role": "user", "content": f"Current observation: {step_result['observation']}"})
            # print(f"messages: {messages[-1]}")

            current_obs = step_result['observation']
            done = step_result['done']
        
            if round >= MAX_ROUND:
                break


        exceeded_max_round = round >= MAX_ROUND
        all_tasks_info.append({
            "task_id": task_count,
            "messages": messages,
            "history": task_history,
            "final_score": step_result['score'][0] if step_result else 0,
            "completed": True if done else False,
            "exceeded_max_round": exceeded_max_round
        })

        print(f"\nTask {task_count} completed or reset.")

        results.append({
            "task_id": task_count,
            "final_score": step_result['score'][0] if step_result else 0,
            "completed": True if done else False,
            "exceeded_max_round": exceeded_max_round
        })

        # if task_count >= chat_model.alfworld_model.env.num_games:
        #     break
        # if task_count >= len(chat_model.alfworld_model.env.gamefiles):
        #     break
        if task_count >= 60:
            break

    success_rate = sum(1 for r in results if r['completed']) / len(results)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all tasks' information
    all_tasks_file = f"all_tasks_{chat_model.alfworld_model.train_eval}_{args.template}_{current_time}.json"
    all_tasks_save_path = os.path.join(chat_model.save_path, all_tasks_file)
    os.makedirs(os.path.dirname(all_tasks_save_path), exist_ok=True)
    with open(all_tasks_save_path, 'w') as f:
        json.dump(all_tasks_info, f, indent=4)
    print(f"All tasks' information saved to {all_tasks_save_path}")

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
    for item in all_tasks_info:
        if item['completed']:
            success_length.append(len(item['history']))
    print(f"Average completed steps: {np.mean(success_length)}")



if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_chat(args))
