import base64
import datetime
import json
import os
import time
from abc import ABC, abstractmethod
from io import BytesIO

from openai import OpenAI
from PIL import Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Logger / Recorder
class ChatRecorder:
    def __init__(self, run_id=None, base_dir="logs"):
        if run_id is None:
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_id = run_id
        self.log_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.log_dir, exist_ok=True)

        self.history_file = os.path.join(self.log_dir, "chat_history.json")
        print(f"Chat session logs saved to: {self.log_dir}")

    def save_history(self, messages):
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: failed to save chat history: {e}")

    def load_history(self):
        if os.path.exists(path=self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: failed to load chat history: {e}")
        return []

    def trim_history_to_step(self, step_num):
        if not os.path.exists(self.history_file):
            print(f"Warning: chat history file not found: {self.history_file}")
            return

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                messages = json.load(f)

            messages_to_keep = []
            step_count = 0
            target_steps = step_num - 1 

            i = 0
            while i < len(messages) and step_count < target_steps:
                if i + 2 >= len(messages):
                    break

                system_msg = messages[i]
                user_msg = messages[i + 1]
                assistant_msg = messages[i + 2]

                is_verify = False
                if system_msg.get("role") == "system":
                    for item in system_msg.get("content", []):
                        if (
                            item.get("type") == "text"
                            and "You are a scene verification agent."
                            in item.get("text", "")
                        ):
                            is_verify = True
                            break

                if is_verify:
                    i += 3
                else:
                    messages_to_keep.extend([system_msg, user_msg, assistant_msg])
                    step_count += 1
                    i += 3

            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(messages_to_keep, f, indent=2, default=str)

            print(f"Trimmed chat_history.json to keep steps 1-{step_count}")
        except Exception as e:
            print(f"Warning: failed to trim chat history: {e}")


class BaseLLM(ABC):
    def __init__(self, recorder: ChatRecorder):
        self.recorder = recorder

    @abstractmethod
    def chat(self, user_text=None, image_path=None, system_text=None):
        pass

    @abstractmethod
    def reload_history(self):
        pass


class OpenAILLM(BaseLLM):
    def __init__(self, base_url, api_key, model_name, recorder=None, history=None):
        super().__init__(recorder)
        self.client = OpenAI(
            base_url=base_url if base_url else "",
            api_key=api_key if api_key else ""
        )
        self.model_name = model_name if model_name else "gemini-3-flash-preview"
        self.json_history = history if history else []

    def chat(self, user_text=None, image_path=None, system_text=None):
        api_messages = []
        
        sys_prompt = system_text if system_text else "You are a helpful assistant."
        api_messages.append({
            "role": "system",
            "content": sys_prompt
        })

        user_content = []
        if image_path:
            if isinstance(image_path, Image.Image):
                buffered = BytesIO()
                image_path.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })
            else:
                if isinstance(image_path, str):
                    image_path = [image_path]
                for path in image_path:
                    image_base64 = encode_image(path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        if user_text:
            user_content.append({"type": "text", "text": user_text})
        
        api_messages.append({"role": "user", "content": user_content})

        max_retries = 3
        retry_delay = 2

        ai_response = ""
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=api_messages,
                    stream=False
                )
                if type(completion) == str:
                    ai_response = json.loads(completion)["choices"][0]["message"]["content"]
                else:
                    ai_response = completion.choices[0].message.content
                break
            except Exception as e:
                print(f"Warning: API request {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
        
        if self.recorder is not None:
            self.json_history.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
            self.json_history.append({"role": "user", "content": user_content})
            self.json_history.append({"role": "assistant", "content": ai_response})
            self.recorder.save_history(self.json_history)

        return ai_response

    def reload_history(self):
        loaded_history = self.recorder.load_history()
        if loaded_history:
            self.json_history = loaded_history


class LLM:
    def __init__(self, model="gemini", run_id=None, resume=False, log=False, log_dir="logs"):
        self.model = model
        if log:
            self.recorder = ChatRecorder(run_id=run_id, base_dir=log_dir)
        else:
            self.recorder = None

        history = []
        if resume and log:
            history = self.recorder.load_history()

        if self.model == "gemini":
            self.delegate = OpenAILLM(
                base_url=os.getenv("API_URL"),
                api_key=os.getenv("API_KEY"),
                model_name="gemini-3-flash-preview",
                recorder=self.recorder,
                history=history,
            )
        elif self.model == "gpt":
            self.delegate = OpenAILLM(
                base_url=os.getenv("API_URL"),
                api_key=os.getenv("API_KEY"),
                model_name="gpt-5.2",
                recorder=self.recorder,
                history=history,
            )
        elif self.model == "qwen":
            self.delegate = OpenAILLM(
                base_url=os.getenv("API_URL"),
                api_key=os.getenv("API_KEY"),
                model_name="qwen3.8max",
                recorder=self.recorder,
                history=history,
            )
        elif self.model == "gpt_external":
            self.delegate = OpenAILLM(
                api_key=os.getenv("API_KEY"),
                base_url=os.getenv("API_URL"),
                model_name="gpt-5.5",
                recorder=self.recorder,
                history=history,
            )
        elif self.model == "claude":
            self.delegate = OpenAILLM(
                api_key=os.getenv("API_KEY"),
                base_url=os.getenv("API_URL"),
                model_name="claude-opus-4-8",
                recorder=self.recorder,
                history=history,
            )
        elif self.model == "ds":
            self.delegate = OpenAILLM(
                api_key=os.getenv("API_KEY"),
                base_url=os.getenv("API_URL"),
                model_name="deepseek-v4-pro",
                recorder=self.recorder,
                history=history,
            )
        else:
            raise ValueError(f"Model {self.model} not supported")

    def chat(self, user_text=None, image_path=None, system_text=None):
        return self.delegate.chat(
            user_text=user_text, image_path=image_path, system_text=system_text
        )

    def reload_history(self):
        self.delegate.reload_history()


if __name__ == "__main__":
    llm = LLM(model="gemini", resume=False)
    response = llm.chat(user_text="Hello, can you help me?")
    print("AI response:", response)
