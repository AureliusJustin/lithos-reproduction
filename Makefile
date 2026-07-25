CUDA   ?= /usr/local/cuda
NVCC   ?= $(CUDA)/bin/nvcc
CC     ?= gcc
CXX    ?= g++
ARCH   ?= native          # auto-detect the installed GPU (A6000=sm_86, A100=sm_80, ...)

CFLAGS  := -Wall -Wno-parentheses -fPIC -O2 -g -Isrc -I$(CUDA)/include
LDFLAGS := -ldl -lpthread -L$(CUDA)/lib64 -lcuda -l:libzstd.so.1

BUILD := build

.PHONY: all clean tests interpose-lib

all: $(BUILD)/liblithos_full.so $(BUILD)/libcuda.so.1 tests

# ----------------------------------------------------------------------
#  LibLithOS interposition library (milestone 1: stub atomizer)
# ----------------------------------------------------------------------
#  LithOS: interposition + TPC scheduler + Kernel Atomizer.
#  The Prelude is JIT-compiled at runtime via NVRTC (address baked in), so no
#  prelude.o is linked; prelude.cu is retained as documentation.
# ----------------------------------------------------------------------
FULL_SRC := src/interpose.c src/real.c src/config.c src/sched.c \
            src/atomizer.c src/atomize_splice.c src/fatbin.c src/qmd.c \
            src/graphsched.c src/predict.c

# liblithos_full.so: LD_PRELOAD in front of a driver-API app.
$(BUILD)/liblithos_full.so: $(FULL_SRC) src/*.h | $(BUILD)
	$(CC) -shared -o $@ $(FULL_SRC) $(CFLAGS) $(LDFLAGS) -lnvrtc

# libcuda.so.1 wrapper: transparent to the CUDA runtime (PyTorch/TF/JAX).
# Patch the NEEDED entry from libcuda.so.1 -> libcuda.so so forwarded symbols
# resolve to the real driver rather than recursing into us (as in libsmctrl).
$(BUILD)/libcuda.so.1: $(FULL_SRC) src/wrapper.c src/*.h | $(BUILD)
	$(CC) -shared -o $@ $(FULL_SRC) src/wrapper.c -DLITHOS_WRAPPER $(CFLAGS) $(LDFLAGS) -lnvrtc
	sed -i "s/libcuda.so.1\x00/libcuda.so\x00\x00\x00/g" $@

# ----------------------------------------------------------------------
#  Tests
# ----------------------------------------------------------------------
tests: $(BUILD)/test_interpose $(BUILD)/test_interpose_driver \
       $(BUILD)/test_atomize_small $(BUILD)/test_atomize_cubin \
       $(BUILD)/test_atomize_runtime $(BUILD)/test_atomize_graph \
       $(BUILD)/atomize_mark.cubin $(BUILD)/atomize_mark.fatbin \
       $(BUILD)/atomize_mark.ptx $(BUILD)/test_scheduler \
       $(BUILD)/correctness_matrix

# CUDA-runtime apps (exercise the libcuda.so.1 wrapper path used by frameworks)
$(BUILD)/test_interpose: tests/test_interpose.cu | $(BUILD)
	$(NVCC) -arch=$(ARCH) $< -o $@ -lcuda
$(BUILD)/test_atomize_runtime: tests/test_atomize_runtime.cu | $(BUILD)
	$(NVCC) -arch=$(ARCH) $< -o $@ -lcudart
$(BUILD)/test_atomize_graph: tests/test_atomize_graph.cu | $(BUILD)
	$(NVCC) -arch=$(ARCH) $< -o $@ -lcudart
# Correctness matrix: varied kernel patterns (elementwise / tiled-matmul-with-
# shared-mem / atomics / multi-dim grids / transpose) each checked vs a CPU
# reference; the atomic kernels catch any double- or missing-block execution.
$(BUILD)/correctness_matrix: tests/correctness/matrix.cu | $(BUILD)
	$(NVCC) -arch=$(ARCH) -O2 $< -o $@ -lcudart

# Driver-API apps (exercise the LD_PRELOAD path)
$(BUILD)/test_interpose_driver: tests/test_interpose_driver.c | $(BUILD)
	$(CC) $< -o $@ -I$(CUDA)/include -L$(CUDA)/lib64 -lcuda
$(BUILD)/test_atomize_small: tests/test_atomize_small.c | $(BUILD)
	$(CC) $< -o $@ -I$(CUDA)/include -L$(CUDA)/lib64 -lcuda
# Real CUBIN / compressed FATBIN / PTX of the same kernel: the atomizer unwraps
# each to a raw cubin, splices it, and atomizes -- exercised by test_atomize_cubin.
$(BUILD)/atomize_mark.cubin: tests/atomize_mark.cu | $(BUILD)
	$(NVCC) -arch=$(ARCH) -cubin $< -o $@
$(BUILD)/atomize_mark.fatbin: tests/atomize_mark.cu | $(BUILD)
	$(NVCC) -arch=$(ARCH) -fatbin -Xfatbin=-compress-all $< -o $@
$(BUILD)/atomize_mark.ptx: tests/atomize_mark.cu | $(BUILD)
	$(NVCC) -arch=$(ARCH) -ptx $< -o $@
$(BUILD)/test_atomize_cubin: tests/test_atomize_cubin.c | $(BUILD)
	$(CC) $< -o $@ -I$(CUDA)/include -L$(CUDA)/lib64 -lcuda
$(BUILD)/test_scheduler: tests/test_scheduler.c | $(BUILD)
	$(CC) $< -o $@ -I$(CUDA)/include -L$(CUDA)/lib64 -lcuda

.PHONY: run_tests
run_tests: all
	@echo "== interposition (driver API / LD_PRELOAD) =="
	LD_PRELOAD=$(BUILD)/liblithos_full.so $(BUILD)/test_interpose_driver
	@echo "== interposition (CUDA runtime / wrapper) =="
	LD_LIBRARY_PATH=$(BUILD) $(BUILD)/test_interpose
	@echo "== Kernel Atomizer -- CUBIN (splice: real atoms, fall-through, no transfer) =="
	LITHOS_ATOM_US=8 LD_PRELOAD=$(BUILD)/liblithos_full.so $(BUILD)/test_atomize_cubin 256 $(BUILD)/atomize_mark.cubin
	@echo "== Kernel Atomizer -- compressed FATBIN (LZ4-decode + unwrap + splice) =="
	LITHOS_ATOM_US=8 LD_PRELOAD=$(BUILD)/liblithos_full.so $(BUILD)/test_atomize_cubin 256 $(BUILD)/atomize_mark.fatbin
	@echo "== Kernel Atomizer -- PTX (JIT to cubin + splice) =="
	LITHOS_ATOM_US=8 LD_PRELOAD=$(BUILD)/liblithos_full.so $(BUILD)/test_atomize_cubin 256 $(BUILD)/atomize_mark.ptx
	@echo "== Kernel Atomizer -- CUDA RUNTIME API path (frameworks: PyTorch/TF/JAX/TensorRT) =="
	LITHOS_ATOM_US=8 LD_LIBRARY_PATH=$(BUILD) $(BUILD)/test_atomize_runtime 256
	@echo "== Kernel Atomizer -- CUDA GRAPH capture -> atomized subgraph (replay) =="
	LITHOS_ATOM_US=8 LD_LIBRARY_PATH=$(BUILD) $(BUILD)/test_atomize_graph 512
	@echo "== TPC scheduler compute quota (4 TPCs) =="
	LITHOS_QUOTA=4 LITHOS_STEALING=0 LD_PRELOAD=$(BUILD)/liblithos_full.so $(BUILD)/test_scheduler
	@echo "== Correctness matrix -- varied kernels vs CPU reference, forced max-split =="
	LITHOS_ATOM_US=1 LITHOS_STATS=1 LD_LIBRARY_PATH=$(BUILD) $(BUILD)/correctness_matrix

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
