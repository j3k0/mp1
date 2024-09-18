import argparse
import sys
import logging
from dotenv import load_dotenv
from utils import generate_response, load_env_vars, setup_logging
from api_handlers import OllamaHandler, GroqHandler, PerplexityHandler  # Updated import

def output_response(args, text):
    print(text, flush=True)
    if args.output:
        with open(args.output, 'a') as f:
            f.write(text)

def main():
    # Load environment variables
    load_dotenv()
    env_vars = load_env_vars()

    parser = argparse.ArgumentParser(description="Generate responses using AI models")
    parser.add_argument("prompt", help="The prompt to generate a response for")
    parser.add_argument("--api", choices=["ollama", "groq", "perplexity"], default="ollama", help="API to use (default: ollama)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--temperature", type=float, default=0.7, help="Set the temperature for the model")
    parser.add_argument("--max-tokens", type=int, default=None, help="Set the maximum number of tokens to generate")
    parser.add_argument("-f", "--file", help="Read prompt from a file")
    parser.add_argument("-o", "--output", help="Save output to a file")
    args = parser.parse_args()

    # Set up logging based on verbosity
    if args.verbose:
        setup_logging(level=logging.DEBUG, stream=sys.stderr)
    else:
        setup_logging(level=logging.ERROR, stream=sys.stderr)
    
    if args.file:
        with open(args.file, 'r') as f:
            args.prompt = f.read().strip()

    if args.api == "ollama":
        api_handler = OllamaHandler(env_vars["OLLAMA_URL"], env_vars["OLLAMA_MODEL"])  # Updated class name
    elif args.api == "groq":
        api_handler = GroqHandler(env_vars["GROQ_MODEL"])  # Updated class name
    elif args.api == "perplexity":
        api_handler = PerplexityHandler(env_vars["PERPLEXITY_API_KEY"], env_vars["PERPLEXITY_MODEL"])  # Updated class name
    else:
        logging.error(f"Invalid API choice: {args.api}")
        sys.exit(1)

    try:
        for steps, total_time in generate_response(args.prompt, api_handler):
            final_step = steps[-1]
            if total_time is None:
                # Intermediate steps
                logging.debug(f"{final_step[0]} (Thinking time: {final_step[2]:.2f}s)")
                output_response(args, final_step[0])
                output_response(args, final_step[1])
                logging.debug("")
            else:
                # Final answer
                output_response(args, 'Answer:')
                output_response(args, final_step[1])
                logging.debug(f"\nTotal thinking time: {total_time:.2f}s")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()