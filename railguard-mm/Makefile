SHELL := /bin/bash

.PHONY: demo showcase-assets showcase-plan open-data test native native-test native-sanitize-test benchmark firmware-host-test firmware-sanitize-test validate integration-smoke up down

demo:
	python scripts/generate_demo_data.py --out data/sample/telemetry.jsonl --minutes 5

showcase-assets:
	python scripts/generate_showcase_assets.py --out-dir docs/assets

open-data:
	python ml/open_data.py list

showcase-plan:
	@test -n "$(TABLE)" || (echo "usage: make showcase-plan TABLE=data/processed/rail_vivid_multirun.csv" && exit 2)
	python ml/run_showcase_experiments.py $(TABLE) --dry-run

test:
	pytest -q

native:
	cmake -S edge/cpp -B build/edge-cpp -DCMAKE_BUILD_TYPE=Release
	cmake --build build/edge-cpp -j2

native-test: native
	./build/edge-cpp/railguard_edge
	./build/edge-cpp/railguard_bench 20000

native-sanitize-test:
	cmake -S edge/cpp -B build/edge-cpp-sanitize -DCMAKE_BUILD_TYPE=RelWithDebInfo \
		-DRAILGUARD_WARNINGS_AS_ERRORS=ON -DRAILGUARD_ENABLE_SANITIZERS=ON -DRAILGUARD_BUILD_BENCHMARKS=OFF
	cmake --build build/edge-cpp-sanitize -j2
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
		ctest --test-dir build/edge-cpp-sanitize --output-on-failure

benchmark: native
	python benchmarks/run_edge_benchmarks.py --iterations 200000 --repeats 7 --out benchmarks/results/latest.json

validate:
	./scripts/validate_all.sh

firmware-host-test:
	mkdir -p build/firmware-host
	gcc -std=c17 -O2 -Wall -Wextra -Werror -Ifirmware/stm32_hal/Core/Inc \
		firmware/stm32_hal/Core/Src/railguard_packet.c firmware/stm32_hal/Core/Src/gnss_time.c firmware/stm32_hal/Core/Src/gnss_stream.c \
		firmware/stm32_hal/Core/Src/dsp_features.c firmware/stm32_hal/Core/Src/environment_decode.c \
		firmware/stm32_hal/Core/Src/transport_queue.c firmware/stm32_hal/host_test.c -lm -o build/firmware-host/host_test
	./build/firmware-host/host_test

firmware-sanitize-test:
	mkdir -p build/firmware-host-sanitize
	gcc -std=c17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer \
		-Ifirmware/stm32_hal/Core/Inc \
		firmware/stm32_hal/Core/Src/railguard_packet.c firmware/stm32_hal/Core/Src/gnss_time.c firmware/stm32_hal/Core/Src/gnss_stream.c \
		firmware/stm32_hal/Core/Src/dsp_features.c firmware/stm32_hal/Core/Src/environment_decode.c \
		firmware/stm32_hal/Core/Src/transport_queue.c firmware/stm32_hal/host_test.c -lm -o build/firmware-host-sanitize/host_test
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ./build/firmware-host-sanitize/host_test

integration-smoke:
	@set -euo pipefail; \
	cp .env.example .env; \
	docker compose config -q; \
	trap 'docker compose down -v' EXIT; \
	docker compose up -d --build timescaledb mosquitto minio ingestor api; \
	python scripts/integration_smoke.py --timeout 120

up:
	docker compose up --build

down:
	docker compose down -v
