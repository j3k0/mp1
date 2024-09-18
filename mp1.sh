#!/bin/bash

# Default values
API="groq"
NUM_RUNS=3
VERBOSE=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api)
            API="$2"
            shift 2
            ;;
        --num_runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -v)
            VERBOSE=1
            shift
            ;;
        -vv)
            VERBOSE=2
            shift
            ;;
        *)
            PROMPT="$@"
            break
            ;;
    esac
done

# Check if prompt is provided
if [ -z "$PROMPT" ]; then
    echo "Usage: $0 [--api <api>] [--num_runs <number_of_runs>] [-v|-vv] <PROMPT>"
    exit 1
fi

run_generate_response() {
    local output_file="output_$1.txt"
    local verbose_flag=""
    
    if [ $VERBOSE -eq 2 ]; then
        verbose_flag="--verbose"
    fi


    if [ $VERBOSE -gt 0 ]; then
        # make sure the output file is empty
        rm -f "$output_file"
        python generate_response.py --api "$API" $verbose_flag "$PROMPT" | tee "$output_file"
    else
        python generate_response.py --api "$API" $verbose_flag "$PROMPT" > "$output_file"
    fi
}

# Run generate_response.py in parallel
for i in $(seq 1 $NUM_RUNS); do
    run_generate_response $i &
done

# Wait for all background jobs to finish
wait

# Combine all outputs
echo "$PROMPT" > combined_output.txt
echo "MULT1 EXPERTS" >> combined_output.txt
for i in $(seq 1 $NUM_RUNS); do
    cat "output_$i.txt" >> combined_output.txt
    echo "---" >> combined_output.txt
    rm "output_$i.txt"
done

# Run final_answer.py
if [ $VERBOSE -gt 0 ]; then
    cat combined_output.txt | python final_answer.py --api "$API" | tee final_output.txt
else
    cat combined_output.txt | python final_answer.py --api "$API" > final_output.txt
fi

# Display final output
cat final_output.txt

# Clean up
rm combined_output.txt final_output.txt
