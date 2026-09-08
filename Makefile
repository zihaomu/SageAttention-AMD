ROCM_PATH ?= /opt/rocm
HIP_ARCHS ?= gfx1201
CXX := $(ROCM_PATH)/bin/hipcc
HIP_CLANG := $(ROCM_PATH)/llvm/bin/clang++
HIP_FLAGS := $(addprefix --offload-arch=,$(HIP_ARCHS))
SAGE_AMDGPU_FLAGS ?=
PYTHON ?= python3
CPPFLAGS := -Iinclude -Isrc
CXXFLAGS := -std=c++17 -O3 -g -MMD -MP -Wall -Wextra -Wpedantic \
	-Wshadow -Wconversion -Wno-sign-conversion $(HIP_FLAGS)
LDLIBS := -lamdhip64

BUILD_DIR ?= build
LIB_OBJECTS := $(BUILD_DIR)/sage_attention.o \
	$(BUILD_DIR)/h3_vdn_sage.o \
	$(BUILD_DIR)/h3_vdn_sage_gfx12.o \
	$(BUILD_DIR)/sage_attention_portable_reference.o
LIBRARY := $(BUILD_DIR)/libsageattention_amd.a
TEST_BINS := $(BUILD_DIR)/test_contract $(BUILD_DIR)/test_gpu
BENCH_BINS := $(BUILD_DIR)/bench_h3_vdn $(BUILD_DIR)/bench_interval

.PHONY: all test metadata-check tool-test contract-test gpu-test bench isa clean

all: $(LIBRARY) $(TEST_BINS) $(BENCH_BINS)

metadata-check:
	$(PYTHON) tools/validate_registry.py

tool-test:
	$(PYTHON) tests/test_sagectl.py

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/sage_attention.o: src/sage_attention.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/h3_vdn_sage.o: src/h3_vdn_sage.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/h3_vdn_sage_gfx12.o: src/h3_vdn_sage_gfx12.hip | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SAGE_AMDGPU_FLAGS) -x hip -c $< -o $@

$(BUILD_DIR)/sage_attention_portable_reference.o: \
		src/sage_attention_portable_reference.hip \
		src/sage_attention_portable_reference.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x hip -c $< -o $@

$(LIBRARY): $(LIB_OBJECTS)
	$(AR) rcs $@ $^

$(BUILD_DIR)/test_contract: tests/test_contract.cpp \
		$(LIBRARY)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -L$(BUILD_DIR) -lsageattention_amd \
		$(LDLIBS) -o $@

$(BUILD_DIR)/test_gpu: tests/test_gpu.cpp \
		$(LIBRARY)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -L$(BUILD_DIR) -lsageattention_amd \
		$(LDLIBS) -o $@

$(BUILD_DIR)/bench_h3_vdn: tests/bench_h3_vdn.cpp \
		$(LIBRARY)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -L$(BUILD_DIR) -lsageattention_amd \
		$(LDLIBS) -o $@

$(BUILD_DIR)/bench_interval: tests/bench_interval.cpp \
		$(LIBRARY)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -L$(BUILD_DIR) -lsageattention_amd \
		$(LDLIBS) -o $@

contract-test: $(BUILD_DIR)/test_contract
	$(BUILD_DIR)/test_contract

gpu-test: $(BUILD_DIR)/test_gpu
	@if ! printf '%s' "$(H3_PHYSICAL_GPU)" | grep -Eq '^[0-9]+$$'; then \
		echo "gpu-test requires an explicit numeric H3_PHYSICAL_GPU" >&2; \
		exit 2; \
	fi
	ROCR_VISIBLE_DEVICES=$(H3_PHYSICAL_GPU) HIP_VISIBLE_DEVICES=0 \
		$(BUILD_DIR)/test_gpu

test: contract-test gpu-test

bench: $(BUILD_DIR)/bench_h3_vdn
	@if ! printf '%s' "$(H3_PHYSICAL_GPU)" | grep -Eq '^[0-9]+$$'; then \
		echo "bench requires an explicit numeric H3_PHYSICAL_GPU" >&2; \
		exit 2; \
	fi
	ROCR_VISIBLE_DEVICES=$(H3_PHYSICAL_GPU) HIP_VISIBLE_DEVICES=0 \
		$(BUILD_DIR)/bench_h3_vdn

$(BUILD_DIR)/h3_vdn_sage_gfx12.s: src/h3_vdn_sage_gfx12.hip | $(BUILD_DIR)
	$(HIP_CLANG) $(CPPFLAGS) -std=c++17 -O3 $(HIP_FLAGS) --cuda-device-only \
		$(SAGE_AMDGPU_FLAGS) -S -x hip $< -o $@

isa: $(BUILD_DIR)/h3_vdn_sage_gfx12.s
	grep -E 'v_wmma_(i32_16x16x16_iu8|f32_16x16x16_bf16)' $<

clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d $(BUILD_DIR)/*.a $(BUILD_DIR)/*.s \
		$(TEST_BINS) $(BENCH_BINS)

-include $(BUILD_DIR)/*.d
