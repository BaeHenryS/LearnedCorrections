# Makefile for running burgers_1d_spectral_T.py with parameters from params.json

# Define the script name and output directory
SCRIPT = ./burgers_1d_spectral_T.py
LOWRES_DIR = ../1d-burgers/1d-lowres-test

# Define the simulation directories
SIM_DIRS = $(shell ls -d $(LOWRES_DIR)/sim_*/)

OUTPUT_DIR = ./spectral_train


# Define the number of runs
NUM_RUNS = 100

# Default target
all: run

# Run the Python script for each simulation directory
run:
	@for i in $(shell seq -f "%06g" 0 $(NUM_RUNS)); do \
        mkdir -p $(OUTPUT_DIR)/sim_$$i; \
        echo "Running simulation in $(OUTPUT_DIR)/sim_$$i..."; \
        python $(SCRIPT) -parameters $(LOWRES_DIR)/sim_$$i/params.json -output $(OUTPUT_DIR)/sim_$$i -seed $$i;\
    done

# Clean the output directory
clean:
	rm -rf $(OUTPUT_DIR)

.PHONY: all run clean