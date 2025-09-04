# LLM BENCHMARKING with Accurate Input and Output Token Throughput
from locust import HttpUser, task, between, events, LoadTestShape # type: ignore
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import uuid
import time
import statistics
import random
import subprocess
import csv
import os
import dotenv
dotenv.load_dotenv()
from datetime import datetime

# Initialize
model_name = "meta/llama-4-maverick-instruct"
# model_name = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
tokenizer = AutoTokenizer.from_pretrained(
    "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-quantized.w4a16",
    # "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    token=os.getenv("HF_TOKEN")
)

concurrent_users=100
prompts = """
hello!
"""

# Store metrics globally
ttft_times = []  
end_to_end_latencies = []  
inter_token_latencies = []  
tokens_per_second_list = []  

# Global variables for throughput
total_input_tokens = 0  
total_output_tokens = 0  
start_benchmark_time = None  # Track when the benchmark starts

class llm_model(HttpUser):
    # Set wait time between tasks
    wait_time = between(0.5, 5)

    @task
    def generate_response(self):
        global total_input_tokens, total_output_tokens

        # Track request start and first token time
        start_time = time.time()
        first_token_time = None
        tokens = []  # Collect output tokens

        # Select a random prompt from Banking77 and append a UUID for uniqueness
        input_text = f"{prompts} {uuid.uuid4()}"
        # input_text = random.choice(prompts)
        
        # Tokenize input and calculate the number of input tokens
        input_length = len(tokenizer(input_text)['input_ids'])
        total_input_tokens += input_length  # Accumulate input tokens across users

        # Send request to the model
        response = self.client.post(
            url="/v1/chat/completions",
            headers={
                'Content-type': 'application/json',
                'Accept': 'application/json',
                'Authorization': 'Bearer ' + os.getenv("DEKALLM_API_KEY")
            },
            data=json.dumps({
                "model": model_name,
                "messages": [{"role": "user", "content": input_text}],
                "stream": True,
                "temperature": 0.9,
                "top_p": 0.9,
                "max_tokens": 128,
                "min_tokens": 20
            }),
            stream=True
        )

        # Process streamed response to capture TTFT and tokens
        for line in response.iter_lines():
            if line:
                token_time = time.time()  # Capture token arrival time
                if first_token_time is None:
                    first_token_time = token_time
                    ttft = (first_token_time - start_time) * 1000  # Convert to ms
                    ttft_times.append(ttft)  # Store TTFT
                tokens.append(line)  # Track each streamed token

        # Track request end time and calculate latency
        end_time = time.time()
        e2e_latency = (end_time - start_time) * 1000  # E2E Latency in ms
        end_to_end_latencies.append(e2e_latency)

        # Calculate the total output tokens from response
        output_length = len(tokens)
        total_output_tokens += output_length  # Accumulate total output tokens

        # Calculate inter-token latency
        if len(tokens) > 1:
            itl = ((end_time - first_token_time) / (len(tokens) - 1)) * 1000  # Convert to ms
            inter_token_latencies.append(itl)

        # Calculate individual user token speed (tokens/sec)
        token_speed = output_length / (end_time - start_time)
        tokens_per_second_list.append(token_speed)

@events.quitting.add_listener
def display_metrics_summary(environment, **kwargs):
    # Ensure the benchmark duration is correctly calculated
    benchmark_duration = time.time() - start_benchmark_time

    # Calculate input and output token throughput (tokens/sec)
    input_token_throughput = total_input_tokens / benchmark_duration
    output_token_throughput = total_output_tokens / benchmark_duration

    # Helper function to calculate statistics
    def calculate_stats(data):
        return {
            "average": round(sum(data) / len(data), 2) if data else 0,
            "maximum": round(max(data), 2) if data else 0,
            "minimum": round(min(data), 2) if data else 0,
            "median": round(statistics.median(data), 2) if data else 0,
        }

    # Calculate stats for each metric
    ttft_stats = calculate_stats(ttft_times)
    e2e_stats = calculate_stats(end_to_end_latencies)
    inter_token_stats = calculate_stats(inter_token_latencies)
    token_speed_stats = calculate_stats(tokens_per_second_list)

    # Print the metrics summary table
    print("\n--- Metrics Summary ---")
    print(f"{'Metric':<40} {'Average':<10} {'Max':<10} {'Min':<10} {'Median':<10}")
    print("-" * 80)
    print(f"{'Time to First Token (ms)':<40} {ttft_stats['average']:<10} {ttft_stats['maximum']:<10} {ttft_stats['minimum']:<10} {ttft_stats['median']:<10}")
    print(f"{'End-to-End Latency (ms)':<40} {e2e_stats['average']:<10} {e2e_stats['maximum']:<10} {e2e_stats['minimum']:<10} {e2e_stats['median']:<10}")
    print(f"{'Inter-Token Latency (ms)':<40} {inter_token_stats['average']:<10} {inter_token_stats['maximum']:<10} {inter_token_stats['minimum']:<10} {inter_token_stats['median']:<10}")
    print(f"{'Individual User Token Speed (tokens/sec)':<40} {token_speed_stats['average']:<10} {token_speed_stats['maximum']:<10} {token_speed_stats['minimum']:<10} {token_speed_stats['median']:<10}")
    print(f"{'Input Token Throughput (tokens/sec)':<40} {round(input_token_throughput, 2):<10}")
    print(f"{'Output Token Throughput (tokens/sec)':<40} {round(output_token_throughput, 2):<10}")
    print("-" * 80)

# Define the load test shape
class StagesShape(LoadTestShape):
    """
    Fixed staged load pattern that runs through predefined stages sequentially
    """
    stages = [
        {"duration": 60, "users": concurrent_users, "spawn_rate": concurrent_users},
        # {"duration": 60, "users": 28, "spawn_rate": 16},
        # {"duration": 60, "users": 60, "spawn_rate": 32},s
        # {"duration": 60, "users": 100, "spawn_rate": 40},
    ]

    def tick(self):
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])

        return None

# Start the benchmark
start_benchmark_time = time.time()