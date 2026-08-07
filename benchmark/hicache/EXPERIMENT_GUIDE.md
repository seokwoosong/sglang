# HiCache Copy Optimization 실험 가이드

이 문서는 unified-memory HiCache copy 최적화(staging buffer 제거)를 실험하는 방법을 설명합니다.

## 1. 최적화 요약

| 항목 | Before (unified_old) | After (unified_new) |
|------|---------------------|---------------------|
| D2H (write-back) | staging buffer relayout → DMA | JIT kernel 직접 gather (staging 없음) |
| H2D (load) | index_select + index_copy_ | JIT kernel 직접 scatter |
| GPU 메모리 | staging buffer 추가 필요 | 불필요 |
| 코드 경로 | 복잡 (3단계) | 단순 (1단계) |

## 2. 실험 환경 요구사항

- NVIDIA GPU (CUDA 지원)
- SGLang 설치 (`pip install -e .`)
- 테스트할 모델 (예: Qwen/Qwen3.5-32B, meta-llama/Llama-3.1-8B-Instruct)
- `aiohttp` 설치 (`pip install aiohttp`)

## 3. 실험 단계

### 3-1. CPU Unit Test (GPU 불필요)

strided tensor 처리가 올바른지 검증합니다:

```bash
cd /path/to/sglang
python -m unittest test.unittest.test_hicache_strided_copy -v
```

**예상 결과**: 8개 테스트 모두 OK

### 3-2. 마이크로벤치마크 (GPU 필요)

D2H/H2D copy 시간을 토큰 수별로 측정합니다:

```bash
# static (unified-memory OFF)
python benchmark/hicache/bench_hicache_microbench.py \
    --version static \
    --gpu cuda:0 \
    --num-slots 4096 \
    --layer-num 48 \
    --head-num 32 \
    --head-dim 128 \
    --output results/microbench_static.json

# unified_old (staging buffer 경유)
python benchmark/hicache/bench_hicache_microbench.py \
    --version unified_old \
    --gpu cuda:0 \
    --output results/microbench_unified_old.json

# unified_new (최적화 — 직접 JIT kernel)
python benchmark/hicache/bench_hicache_microbench.py \
    --version unified_new \
    --gpu cuda:0 \
    --output results/microbench_unified_new.json
```

**측정 항목**:
- D2H: GPU → Host copy 시간 (μs)
- H2D: Host → GPU copy 시간 (μs)
- 토큰 수: 1, 16, 64, 256, 1024, 4096

### 3-3. End-to-End 서빙 벤치마크 (GPU 필요)

실제 서빙 환경에서 throughput/latency를 측정합니다.

#### Step 1: 서버 시작

```bash
# 최적화 버전 (unified_new)
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-32B \
    --port 30000 \
    --enable-unified-memory \
    --hicache-ratio 0.5 \
    --trust-remote-code
```

#### Step 2: 벤치마크 실행

```bash
# Short prefix (빈번한 eviction)
python benchmark/hicache/bench_hicache_serving.py \
    --port 30000 \
    --scenario short_prefix \
    --output results/serving_short_prefix.json

# Long prefix (빈번한 hit)
python benchmark/hicache/bench_hicache_serving.py \
    --port 30000 \
    --scenario long_prefix \
    --output results/serving_long_prefix.json

# Mixed (현실적 워크로드)
python benchmark/hicache/bench_hicache_serving.py \
    --port 30000 \
    --scenario mixed \
    --output results/serving_mixed.json
```

#### Step 3: 서버 종료 후 다른 버전으로 반복

```bash
# 서버 종료 (Ctrl+C)

# static 버전 (unified-memory OFF)
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-32B \
    --port 30000 \
    --hicache-ratio 0.5 \
    --trust-remote-code
# (같은 벤치마크 반복)
```

### 3-4. 전체 자동화 스크립트

모든 실험을 한 번에 실행:

```bash
bash benchmark/hicache/bench_hicache_copy_optimization.sh Qwen/Qwen3.5-32B cuda:0
```

### 3-5. 리포트 생성

```bash
python benchmark/hicache/generate_report.py \
    --input benchmark/hicache/results \
    --timestamp <TIMESTAMP>
```

## 4. nsys 프로파일링

GPU 커널 실행 패턴을 분석:

```bash
# nsys 프로파일링과 함께 서버 시작
nsys profile \
    --output nsys_unified_new \
    --force-overwrite true \
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3.5-32B \
        --port 30000 \
        --enable-unified-memory \
        --hicache-ratio 0.5 \
        --trust-remote-code

# 다른 터미널에서 워크로드 실행
python benchmark/hicache/bench_hicache_serving.py \
    --port 30000 \
    --scenario long_prefix \
    --num-requests 50

# 서버 종료 후 nsys 리포트 확인
nsys-ui nsys_unified_new.nsys-rep
```

**확인 포인트**:
- `hicache_transfer_per_layer` 커널 호출 횟수
- staging buffer 관련 커널이 사라졌는지 확인
- D2H/H2D DMA 전송 시간

## 5. 기대 효과

| 지표 | 예상 개선 |
|------|----------|
| D2H copy 시간 | 20-40% 감소 (staging 단계 제거) |
| H2D copy 시간 | 10-30% 감소 (직접 scatter) |
| GPU 메모리 | staging buffer만큼 절약 |
| Throughput | eviction 빈도가 높을수록 큰 개선 |

## 6. 트러블슈팅

### JIT 커널 로드 실패
```
WARNING: Failed to load JIT HiCache kernel
```
→ `element_size % 128 != 0`인지 확인. head_num * head_dim * dtype.itemsize가 128의 배수여야 함.

### strided tensor .view() 에러
```
RuntimeError: view size is not compatible with input tensor's size and stride
```
→ `_to_2d_view()`가 자동으로 `as_strided` fallback을 사용하므로 정상 동작.

### 서버 시작 실패
→ `--enable-unified-memory` 플래그가 있는지 확인.
→ GPU 메모리 부족 시 `--hicache-ratio`를 낮춤 (예: 0.3).
