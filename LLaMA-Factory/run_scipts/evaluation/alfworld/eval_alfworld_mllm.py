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
import cv2    
import time
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import requests
import torch
from PIL import Image
# from transformers import MllamaForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import get_peft_model, PeftModel
from difflib import SequenceMatcher

MAX_ROUND = 25
SYSTEM_PROMPT = f"""
You are an AI robot agent in an interactive environment. Your goal is to accomplish the given task through a series of actions. Follow these guidelines:
1. Carefully analyze the task requirements.When searching for required objects, use common sense to predict likely locations.
2. For tasks involving multiple objects, keep a mental count of how many you've collected or placed. For tasks requiring multiple identical items (such as "put two items"), ensure that you actually find and place two different items.
3. Respond only with a JSON object containing exactly these two fields:
    - "what_you_see": your detailed observation
    - "action": one action name from the provided candidates
Any response that is not in this exact JSON format will be considered invalid.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Run ALFWorld with ChatModel")
    parser.add_argument("--config_file", type=str, default="/mnt/tangyehui/code/LLaMA-Factory/evaluation/alfworld/base_config_v1.yaml", help="Path to the config file")
    parser.add_argument("--model_name_or_path", type=str,  default="/mnt/tangyehui/model/llama_32_11b_instruct", help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--adapter_name_or_path", type=str, help="Path to the adapter (LoRA) weights",default = '')
    parser.add_argument("--finetuning_type", type=str, default="lora", help="Finetuning type (e.g., 'lora')")
    parser.add_argument("--template", type=str, default="llama3", help="Template for the model")
    parser.add_argument("--save_path", type=str, default="mcts_datagen/test/result/", help="Path to save the results")
    parser.add_argument("--infer_backend", type=str, default="vllm", help="vllm multimodal")
    return parser.parse_args()

def load_alf_config(config_file: str):
    original_argv = sys.argv
    sys.argv = [sys.argv[0], config_file]

    try:
        config = generic.load_config()
    finally:
        sys.argv = original_argv
    
    return config

def save_frame(env, image_dir):
    existing_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    frame_count = len(existing_files) + 1
    frame_filename = f"{frame_count:04d}.png"
    frame_path = os.path.join(image_dir, frame_filename)
    if frame_path:
        images = env.get_frames()
        cv2_image = cv2.cvtColor(images[0], cv2.COLOR_RGB2RGBA)
        cv2.imwrite(frame_path, cv2_image)
    return frame_path

def parse_textworld_objects(text):
    # 找到 "you see" 后面的内容
    objects_part = text.split("you see ")[-1].split(".\n\nYour task is to: ")[0]

    # 将文本分割成单独的物品
    # 使用逗号分割，并移除最后可能的句号或换行符
    objects = objects_part.strip('.\n').split(', ')
    
    # 为每个物品添加 "go to" 前缀
    go_to_list = ["go to " + obj.replace('and ','').replace('a ','') for obj in objects]
    
    return go_to_list

def get_candidate_action_one_object(template, action, admissible_commands):
    pattern = rf"{template} (\w+)\s+(\d+)"
    match = re.search(pattern, action)
    recep = match.group(1)    # 'fridge'
    index = match.group(2)    # '1'
    candidates = [act for act in admissible_commands if f'{template} {recep}' in act]
    if len(candidates) == 0:
        # 不能template这个recep，返回原始动作保持图片不变
        return action
    else:
        # 有 template recep, 对齐index
        best_candidate = [act for act in candidates if index in act]
        if len(best_candidate) ==0:
            return candidates[0]
        else:
            return best_candidate[0]

def get_candidate_action_two_object(template, action, admissible_commands):
    best_candidate = action
    pattern = rf"^{template[0]}\s+(\w+)\s+(\d+)\s+{template[1]}\s+(\w+)\s+(\d+)$"

    match = re.search(pattern, action)
    obj_name = match.group(1)    # 第一个物品名称
    obj_id = match.group(2)      # 第一个物品ID
    target_name = match.group(3)  # 第二个物品名称
    target_id = match.group(4)    # 第二个物品ID

    candidates = [act for act in admissible_commands if f'{template[0]} {obj_name}' in act]
    if len(candidates) != 0:
        best_candidate = candidates[0]
    
        candidates_1 = [act for act in candidates if target_name in act]
        if len(candidates_1) != 0:
            best_candidate = candidates_1[0]

            candidates_2 = [act for act in candidates_1 if obj_id in act or target_id in act]
            if len(candidates_2) != 0:
                best_candidate = candidates_2[0]

                candidates_3 = [act for act in candidates_2 if obj_id in act and target_id in act]
                if len(candidates_3)!=0:
                    best_candidate = candidates_3[0]
    return best_candidate

def find_most_similar_command_advanced(action: str, admissible_commands: List[str]) -> str:
    # action在admissible_commands里
    if action in admissible_commands:
        return action


    one_object_templates = ["go to","open","close","use",]
    for template in one_object_templates:
        if template in action:
            return get_candidate_action_one_object(template, action, admissible_commands)

    two_object_templates = [['take','from'], ['put', 'in/on'], ['heat', 'with'], ['cool', 'with'], ['clean', 'with'], ['slice', 'with']]
    for template in two_object_templates:
        if template[0] in action:
            return get_candidate_action_two_object(template, action, admissible_commands)
    return action

def extract_json(text):
    pattern = r'\{[\s\S]*?\}'
    match = re.search(pattern, text)
    if match:
        json_str = match.group()
        # 将单引号替换为双引号，但要避免处理嵌套的情况
        json_str = json_str.replace("'", '"')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 如果解析失败，尝试手动解析
            items = {}
            # 移除花括号
            content = json_str.strip('{}')
            # 分割键值对
            pairs = content.split(',')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    # 清理空白字符和引号
                    key = key.strip().strip('"').strip("'")
                    value = value.strip().strip('"').strip("'")
                    items[key] = value
            return items
    return None

class ALFWorldChatModel:
    def __init__(self, engine, config: Dict[str, Any], seed: int = 42):
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
        self.env = self.env.init_env(batch_size=1)
        print(f"Environment type: {env_type}")
        # print(f"Number of tasks in json_file_list: {len(self.env.json_file_list)}")
        print(f"Batch size: {self.env.batch_size}")
        self.reset_environment()


    def reset_environment(self, image_dir = None):
        self.image_dir = image_dir
        self.obs, self.info = self.env.reset()
        if image_dir is not None:
            self.image_path =  save_frame(self.env, self.image_dir)
            return self.obs, self.info, self.image_path
        return self.obs, self.info
        
    async def run_step(self, messages: List[Dict[str, str]], admissible_commands: List[str]) -> Dict[str, Any]:
        MAX_ATTEMPTS = 10
        for attempt in range(MAX_ATTEMPTS):
            try:
                responses = await self.engine.chat(messages, SYSTEM_PROMPT)
                responses_json = extract_json(responses)
                action = find_most_similar_command_advanced(responses_json['action'].strip().strip("'"), admissible_commands)
                break  # 如果成功，跳出循环
            except Exception as e:
                if attempt == MAX_ATTEMPTS - 1:  # 如果是最后一次尝试
                    print(f"Failed after {MAX_ATTEMPTS} attempts. Final error: {str(e)}")
                    raise  # 如果全部尝试都失败，抛出最后一个异常
                else:
                    print(f"Attempt {attempt + 1} failed: {str(e)}, responses: {responses}, retrying...")
                    continue
        
        if 'spatul1' in action:
            action = action.replace("spatul1",'spatula 1')# alfworld环境错误，给的candidate action不对
        new_obs, scores, dones, infos = self.env.step([action])

        if "Nothing happens" in new_obs or 'close cabinet' in action:
            if action in messages[-1]['content']:
                messages[-1]['content'].replace(action,'')
            
            # 再来一次
            responses = await self.engine.chat(messages, system)
            responses_json = extract_json(responses)
            action = responses_json[-1].response_text.strip().strip("'")
            print(f"new_action:{action}")

            new_obs, scores, dones, infos = self.env.step([action])
        self.obs = new_obs
        self.info = infos
        self.frame_path = save_frame(self.env, self.image_dir)
        time.sleep(1)

        return {
            "action": action,
            "observation": self.obs[0],
            "score": infos['goal_condition_success_rate'] if 'goal_condition_success_rate' in infos else [0.0],
            "done": dones[0],
            "info": self.info,
            "frame_path": self.frame_path,
            'what_you_see': responses_json['what_you_see'],
            'responses':responses,
            "admissible_commands":infos["admissible_commands"][0],
        }

def _start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class Qwen2VLModel():
    def __init__(self, model_args):

        model_id = model_args['model_name_or_path']
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto',) 
        self.processor = AutoProcessor.from_pretrained(model_id)

        if model_args["adapter_name_or_path"]:
            for adapter_name_or_path in model_args["adapter_name_or_path"]:
                self.model = PeftModel.from_pretrained(self.model, adapter_name_or_path, is_trainable=False,device_map='auto',)

                peft_config = self.model.peft_config
                
                # 可选：合并 LoRA 权重到基础模型
                self.model = self.model.merge_and_unload()
                
                # 可选：打印配置确认
                print("Loaded LoRA config:", self.model.peft_config)
        
    def convert_messages_single(self, messages: List[Dict[str, str]],system: str = ""):
        # single image
        # 转化为llama可以接受的message
        dialog = [{'role':'system','content':[{
                                "type": "text", 
                                "text": SYSTEM_PROMPT}]
                                }]
        for i, sample_dict in enumerate(messages):
            if sample_dict['role'] == 'user':
                # 分两种
                if len(dialog) == 1:
                    # dialog为system
                    if len(messages) == 1:
                        dialog.append(
                            {'role':'user','content':[{"type": "image",},{
                                "type": "text", 
                                "text": f"Init environment: {sample_dict['current_obs']}"
                                }]
                            }
                        )
                    else:
                        dialog.append(
                            {'role':'user','content':[{
                                "type": "text", 
                                "text": f"Init environment: {sample_dict['current_obs']}"
                                }]
                            }
                        )
                else:
                    # 不为空
                    if i != len(messages)-1:
                        # 不为最后一个
                        what_you_see = messages[i+1]['what_you_see']
                        if "You arrive at" in what_you_see:
                            what_you_see = what_you_see.split('.')[1]
                        dialog.append(
                            {
                                'role':'user',
                                'content':[{
                                    'type':'text',
                                    'text':what_you_see
                                }]
                            }
                        )
                    else:
                        # 最后一步
                        dialog.append(
                            {
                                'role':'user',
                                'content':[
                                    {"type": "image"},
                                    {"type": "text", "text": 'Descibe what you see. And Decide the next action.'}]
                            }
                        )
            else:
                dialog.append({"role":"assistant","content":sample_dict['content']})
        image_path = sample_dict['frame_path']
        image = Image.open(image_path)
        return dialog, image


    def convert_messages_multi(self, messages: List[Dict[str, str]],system: str = ""):
        # 转化为llama可以接受的message
        # multi image
        dialog = [{'role':'system','content':[{
                                "type": "text", 
                                "text": SYSTEM_PROMPT}]
                                }]
        dialog_images = []
        for i, sample_dict in enumerate(messages):
            if sample_dict['role'] == 'user':
                if 'Welcome to TextWorld, ALFRED!' in sample_dict['current_obs']:
                    dialog.append({'role':'user',
                    'content':[{"type": "image",},{"type": "text", "text": f"{sample_dict['current_obs']}"}]})
                else:
                    dialog.append({'role':'user',
                    'content':[{"type": "image",},{"type": "text", "text": f"Descibe what you see. And Decide the next action."}]})

                with Image.open(sample_dict["frame_path"]) as img:
                    image = img.convert("RGB")
                
                dialog_images.append(image)

            else:
                dialog.append(
                    {"role":"assistant","content":[{"type": "text", "text": sample_dict['content']}]}
                )
        return dialog, dialog_images

    async def chat(self, messages: List[Dict[str, str]], system: str = "") -> List[Dict[str, Any]]:
        messages, image = self.convert_messages_single(messages, system)
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,)

        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=500)

        # json格式
        # return self.processor.decode(output[0][inputs["input_ids"].shape[-1]:])
        return self.processor.decode(output[0][inputs["input_ids"].shape[-1]:])
        
    async def stream_chat(self, messages: List[Dict[str, str]], system: str = ""):
        raise NotImplementedError("stream_chat is not implemented for GPT4Engine")

    def get_scores(self, texts: List[str], **kwargs) -> Dict[str, List[float]]:
        raise NotImplementedError("get_scores is not implemented for GPT4Engine")

class ChatModel:
    def __init__(self, args: Dict[str, Any]) -> None:
        hf_args = deepcopy(args)
        
        config_file = hf_args.pop('config_file', None)
        save_path = hf_args.pop('save_path', None)

        hf_args['adapter_name_or_path'] = hf_args['adapter_name_or_path'].split(',')
                
        self.engine = Qwen2VLModel(hf_args)

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()        
        
        config = load_alf_config(config_file)
        self.alfworld_model = ALFWorldChatModel(self.engine, config)
        
        self.save_path = save_path

    async def run_alfworld_step(self, messages: List[Dict[str, str]], admissible_commands: List[str]) -> Dict[str, Any]:
        return await self.alfworld_model.run_step(messages ,admissible_commands)

    def reset_alfworld(self, image_dir) -> str:
        return self.alfworld_model.reset_environment(image_dir)

async def run_chat(args: argparse.Namespace) -> None:
    chat_model = ChatModel(vars(args))
    print("Welcome to the ALFWorld CLI. Running tasks automatically.")
    results = []
    all_tasks_info = []
    task_count = 0


    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all success tasks' information

    root_image_dir = os.path.join(chat_model.save_path,'images',f"{chat_model.alfworld_model.train_eval}_{args.template}_{current_time}")
    print(f"Save image in {root_image_dir}")

    while True:
        round = 0
        task_count += 1
        image_dir = os.path.join(root_image_dir,str(task_count))
        os.makedirs(image_dir, exist_ok=True) 
        obs, info, frame_path = chat_model.reset_alfworld(image_dir)
        game_file = info['extra.gamefile'][0]
        
        current_obs = obs[0]
        messages = []
        
        done = False
        task_history = []
        admissible_commands = info["admissible_commands"][0]


        messages.append({"role": "user", "current_obs": current_obs, 'frame_path':frame_path})
        try:
            while not done:
                round += 1

                step_result = await chat_model.run_alfworld_step(messages, admissible_commands)

                if step_result is None:
                    print("Error occurred during step execution. Resetting environment.")
                    break
                
                print(f"LLM Observation {round}: {step_result['responses']}")
                print(f"Action {round}: {step_result['action']}")
                print(f"Observation {round}: {step_result['observation']}")
                print("\n")
                
                task_history.append({
                    "action": step_result['action'],
                    "observation": step_result['observation'],
                    "score": step_result['score'],
                    "done": True if step_result['done'] else False,
                    'frame_path': step_result['frame_path'],
                    'what_you_see': step_result['what_you_see'],
                })
                
                admissible_commands = step_result['admissible_commands']

                current_action_space = [action for action in step_result['info']["admissible_commands"][0] if not action.startswith("look")]

                messages.append({"role": "assistant", "action": step_result['action'], 'what_you_see':step_result['what_you_see'],'content':step_result['responses']})

                messages.append({"role": "user", "current_obs": step_result['what_you_see'],"frame_path":step_result['frame_path']})

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
                "exceeded_max_round": exceeded_max_round,
                "image_dir":image_dir,
            })
            if task_count >= 60 or task_count>=chat_model.alfworld_model.env.num_games: #chat_model.alfworld_model.env.num_games:
                break
        except Exception as e:  # 捕获具体的异常并打印
            print(f"Error in task {task_count}: {str(e)}")
            all_tasks_info.append({
                "task_id": game_file,
                "messages": messages,
                "history": task_history,
                "final_score": 0,
                "completed": False,
                "exceeded_max_round": exceeded_max_round
            })
            
            results.append({
                "task_id": game_file,
                "final_score": 0,
                "completed": False,
                "exceeded_max_round": exceeded_max_round,
                "image_dir":image_dir,
            })


    success_rate = sum(1 for r in results if r['completed']) / len(results)
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