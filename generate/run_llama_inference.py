"""
python generate/run_llama_inference.py --partition_numbers 6 7 --batch_numbers 4 --machine jagupard38
"""


import os
import argparse

def run_commands(partition_numbers, batch_numbers, machine):
    for partition_num in partition_numbers:
        for batch_num in batch_numbers:
            cmd = (
                f"nlprun -g 1 -m {machine} -c 8 -r 80G -a culturebank "
                f"'python ~/CultureBank/lit-gpt_ours/generate/base_chat_filter_culture_ryan.py "
                f"--checkpoint_dir ~/CultureBank/lit-gpt_ours/checkpoints/meta-llama/Llama-2-70b-chat-hf/ "
                f"--precision bf16-true --quantize bnb.nf4-dq "
                f"--comment_dir /sphinx/u/culturebank/tiktok_data/combined_comments/8_partitions/cultural_relevance/partitions/part{partition_num}/part{partition_num}_batch{batch_num}.csv "
                f"--video_dir /juice/scr/weiyans/CultureBank/tiktok_data/videos "
                f"--results_dir /sphinx/u/culturebank/tiktok_results/partitions/part{partition_num}/batch{batch_num}.json "
                f"--temperature 0.01 --top_k 1'"
            )
            os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run commands for partition and batch numbers.")
    parser.add_argument('--partition_numbers', nargs='+', type=int, required=True, help='List of partition numbers')
    parser.add_argument('--batch_numbers', nargs='+', type=int, required=True, help='List of batch numbers')
    parser.add_argument('--machine', type=str, required=True, help='Machine name')

    args = parser.parse_args()

    run_commands(args.partition_numbers, args.batch_numbers, args.machine)

if __name__ == "__main__":
    main()