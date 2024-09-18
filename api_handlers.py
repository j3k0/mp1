import json
import requests
import groq
import time
import re
import logging

temperature = 0.7 # more variability in the response

def set_temperature(temp):
    global temperature
    temperature = temp

# str contains a string that is a json but maybe has been truncated 
def fix_incomplete_json(s):
    # Count open and close brackets/braces
    open_curly = s.count('{')
    close_curly = s.count('}')
    open_square = s.count('[')
    close_square = s.count(']')
    num_quotes = s.count('"')

    # Add missing closing brackets/braces
    if (num_quotes % 2 == 1):  # Changed from len(num_quotes) to num_quotes
        s += '"'
    if (open_square > close_square):
        s += ']' * (open_square - close_square)
    if (open_curly > close_curly):  
        s += '}' * (open_curly - close_curly)

    # Attempt to parse the fixed JSON
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        logging.warning("Warning: content is not a valid JSON: %s", s)
        # If still invalid, return None or original string
        return s  # or return s if you prefer

class OllamaHandler:
    def __init__(self, url, model):
        self.url = url
        self.model = model

    def make_api_call(self, messages, max_tokens, is_final_answer=False, is_json_content=True, temperature=temperature):
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        }
                    }
                )
                response.raise_for_status()
                if is_json_content:
                    return json.loads(response.json()["message"]["content"])
                else:
                    return response.json()["message"]["content"]
            except Exception as e:
                logging.error(e)
                if attempt == 2:
                    return self._error_response(str(e), is_final_answer)
                time.sleep(1)

    def _error_response(self, error_msg, is_final_answer):
        if is_final_answer:
            return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {error_msg}"}
        else:
            return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {error_msg}", "next_action": "final_answer"}

class PerplexityHandler:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    def make_api_call(self, messages, max_tokens, is_final_answer=False, is_json_content=True, temperature=temperature):

        # Quick dirty fix for API calls in perplexity that removes the assistant message
        #messages[0]["content"] = messages[0]["content"] + " You will always respond ONLY with JSON with the following format: {'title': 'Title of the step', 'content': 'Content of the step', 'next_action': 'continue' or 'final_answer'}. You are not allowed to respond with anything else or any additional text. "
        if not is_final_answer:
            for i in range(len(messages)):
                if messages[i]["role"] == "assistant":
                    messages.pop(i)     

        for attempt in range(3):
            try:
                url = "https://api.perplexity.ai/chat/completions"
                payload = {"model": self.model, "messages": messages}
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                response = requests.post(url, json=payload, headers=headers)
                
                # Add specific handling for 400 error
                if response.status_code == 400:
                    error_content = response.json()
                    logging.error("HTTP 400 Error: %s", error_content)
                    return self._error_response(f"HTTP 400 Error: {error_content}", is_final_answer)
                
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                logging.debug("Content: %s", content)
                if is_json_content:
                    return json.loads(content)
                else:
                    return content
            except json.JSONDecodeError:
                logging.warning("Warning: content is not a valid JSON, returning raw response")
                # Better detection of final answer in the raw response for Perplexity
                forced_final_answer = False
                if '"next_action": "final_answer"' in content.lower().strip():
                    forced_final_answer = True
                logging.debug("Forced final answer: %s", forced_final_answer)
                
                return {
                    "title": "Raw Response",
                    "content": content,
                    "next_action": "final_answer" if (is_final_answer|forced_final_answer) else "continue"
                }
            except requests.exceptions.RequestException as e:
                logging.error("Request failed: %s", e)
                if attempt == 2:
                    return self._error_response(str(e), is_final_answer)
                time.sleep(1)

    def _error_response(self, error_msg, is_final_answer):
        return {
            "title": "Error",
            "content": f"API request failed after 3 attempts. Error: {error_msg}",
            "next_action": "final_answer",
        }

class GroqHandler:
    def __init__(self, model="llama-3.1-70b-versatile"):
        self.client = groq.Groq()
        self.model = model

    def make_api_call(self, messages, max_tokens, is_final_answer=False, is_json_content=True, temperature=temperature):
        content = ""
        for attempt in range(3):
            try:
                logging.debug("Messages: %s", messages)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                logging.debug("Response: %s", content)
                logging.debug("Type of response: %s", type(content))
                if is_json_content:
                    return json.loads(fix_incomplete_json(content))
                else:
                    return content
            except groq.BadRequestError as e:
                logging.error("Error in Groq: %s", e)
                error_dict = e.response.json()
                if 'error' in error_dict and 'failed_generation' in error_dict['error']:
                    failed_generation = error_dict['error']['failed_generation']
                    logging.debug("Failed generation: %s", failed_generation)
                    try:
                        if is_json_content:
                            return json.loads(fix_incomplete_json(failed_generation))
                        else:
                            return failed_generation
                    except json.JSONDecodeError:
                        logging.error("Failed to parse failed_generation as JSON: %s", failed_generation)
                        return self._error_response("Failed to parse response as JSON", is_final_answer)
                if attempt == 2:
                    return self._error_response(str(e), is_final_answer)
                time.sleep(1)
            except Exception as e:
                logging.error("Unexpected error in Groq: %s", e)
                if attempt == 2:
                    return self._error_response(str(e), is_final_answer)
                time.sleep(1)

    def _error_response(self, error_msg, is_final_answer):
        if is_final_answer:
            return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {error_msg}"}
        else:
            return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {error_msg}", "next_action": "final_answer"}