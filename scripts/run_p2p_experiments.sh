#!/bin/bash

# Script to run p2p multi-threaded benchmark experiments
# Tests different thread counts and buffer sizes for both sendrecv and isend

# Configuration
BINARY="../build/test_p2p_mthread"
OUTPUT_DIR="./p2p_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Thread counts (powers of 2 from 1 to 128)
THREAD_COUNTS=(1 2 4 8 16 32 64)

# Buffer sizes in bytes
BUFFER_SIZES=(1 1000 1000000 200000000)
BUFFER_LABELS=("1B" "1KB" "1MB" "200MB")

# Methods to test
METHODS=("sendrecv" "isend")

# Create output directory
mkdir -p $OUTPUT_DIR

# Initialize CSV files
SENDRECV_CSV="${OUTPUT_DIR}/sendrecv_results_${TIMESTAMP}.csv"
ISEND_CSV="${OUTPUT_DIR}/isend_results_${TIMESTAMP}.csv"

echo "threads,buffer_size,buffer_label,total_time,avg_time_ms" > $SENDRECV_CSV
echo "threads,buffer_size,buffer_label,total_time,avg_time_ms" > $ISEND_CSV

echo "Starting P2P Multi-threaded Benchmark Experiments"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run experiments
for method in "${METHODS[@]}"; do
    echo "Testing method: $method"
    echo "------------------------"

    # Select the appropriate CSV file
    if [ "$method" == "sendrecv" ]; then
        CSV_FILE=$SENDRECV_CSV
    else
        CSV_FILE=$ISEND_CSV
    fi

    for idx in "${!BUFFER_SIZES[@]}"; do
        buffer_size=${BUFFER_SIZES[$idx]}
        buffer_label=${BUFFER_LABELS[$idx]}

        echo "  Buffer size: $buffer_label ($buffer_size bytes)"

        for threads in "${THREAD_COUNTS[@]}"; do
            echo "    Testing with $threads threads..."

            # Set number of OpenMP threads
            export OMP_NUM_THREADS=$threads

            # Run the benchmark (2 MPI ranks, 1 per node)
            # Capture output
            OUTPUT=$(srun -N 2 -n 2 --ntasks-per-node=1 --gpus-per-task=1 $BINARY $method $buffer_size 2>&1)

            # Parse the output to extract timing information
            # Look for lines like:
            #   Total time: 0.123456 seconds
            #   Average time per operation: 1.234567 ms

            total_time=$(echo "$OUTPUT" | grep "Total time:" | awk '{print $3}')
            avg_time=$(echo "$OUTPUT" | grep "Average time per operation:" | awk '{print $5}')

            # Write to CSV
            echo "$threads,$buffer_size,$buffer_label,$total_time,$avg_time" >> $CSV_FILE

            # Print summary
            echo "      Total time: $total_time s, Avg per op: $avg_time ms"

            # Small delay between runs
            sleep 1
        done
        echo ""
    done
    echo ""
done

echo "Experiments complete!"
echo "Results saved to:"
echo "  SendRecv: $SENDRECV_CSV"
echo "  Isend:    $ISEND_CSV"
echo ""
echo "To plot results, run:"
echo "  python plot_p2p_results.py $SENDRECV_CSV $ISEND_CSV"
