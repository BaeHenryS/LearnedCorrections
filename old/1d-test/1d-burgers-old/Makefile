# Makefile for running burgers_1d.py 100 times

# Define the script name and output directory
SCRIPT = burgers_1d.py
OUTPUT_DIR = 1d-lowres-test

# Define the number of runs -1
NUM_RUNS = 100

# Default target
all: run

# Run the Python script 100 times
run:
	@mkdir -p $(OUTPUT_DIR)
	@for i in $(shell seq 1 $(NUM_RUNS)); do \
		echo "Running iteration $$i..."; \
		python $(SCRIPT) -output $(OUTPUT_DIR) -seed $$i;\
	done

# Clean the output directory
clean:
	rm -rf $(OUTPUT_DIR)

.PHONY: all run clean
