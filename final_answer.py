import argparse
import sys
import logging
from dotenv import load_dotenv
from utils import final_answer, load_env_vars, setup_logging
from api_handlers import OllamaHandler, GroqHandler, PerplexityHandler

def read_input():
    question = ""
    responses = ""
    reading_question = True
    
    for line in sys.stdin:
        line = line.rstrip('\n')  # Remove trailing newline
        if line == "MULT1 EXPERTS":
            reading_question = False
            continue
        
        if reading_question:
            question += line + '\n'
        else:
            responses += line + '\n'
    
    return question.strip(), responses.strip()

def output_response(args, text):
    print(text, flush=True)
    if args.output:
        with open(args.output, 'a') as f:
            f.write(text)

def main():
    # Load environment variables
    load_dotenv()
    env_vars = load_env_vars()

    parser = argparse.ArgumentParser(description="Generate final answer using AI models")
    parser.add_argument("--api", choices=["ollama", "groq", "perplexity"], default="ollama", help="API to use (default: ollama)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--temperature", type=float, default=0.7, help="Set the temperature for the model")
    parser.add_argument("-o", "--output", help="Save output to a file")
    args = parser.parse_args()

    # Set up logging based on verbosity
    if args.verbose:
        setup_logging(level=logging.DEBUG, stream=sys.stderr)
    else:
        setup_logging(level=logging.ERROR, stream=sys.stderr)

    question, responses = read_input()

    if args.api == "ollama":
        api_handler = OllamaHandler(env_vars["OLLAMA_URL"], env_vars["OLLAMA_MODEL"])
    elif args.api == "groq":
        api_handler = GroqHandler(env_vars["GROQ_MODEL"])
    elif args.api == "perplexity":
        api_handler = PerplexityHandler(env_vars["PERPLEXITY_API_KEY"], env_vars["PERPLEXITY_MODEL"])
    else:
        logging.error(f"Invalid API choice: {args.api}")
        sys.exit(1)

    try:
        final_response = final_answer(question, responses, api_handler)
        output_response(args, 'Final Answer:')
        output_response(args, final_response)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()