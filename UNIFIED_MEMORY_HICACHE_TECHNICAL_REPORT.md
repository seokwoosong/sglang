# SGLang Unified Memory와 HiCache의 통합: 설계, 동기화, 검증 및 성능 평가

> 기술 보고서
>
> 대상 브랜치: `feat/unified-memory-hicache`
>
> 기준 parent: `86855d1d1`, 기능 커밋: `99b384342`
> 평가일: 2026-08-04

## 초록

SGLang의 `--enable-unified-memory`는 CUDA Unified Virtual Memory(UVM)를
활성화하는 옵션이 아니다. Hybrid full-attention + Mamba/GDN 모델에서
full-attention KV cache와 Mamba recurrent state가 하나의 GPU byte buffer를
동적으로 나누어 사용하게 하는 SGLang 전용 allocator 기능이다. HiCache는 이
GPU cache를 L1으로 보고 pinned host memory(L2)와 선택적 storage backend(L3)로
확장한다.

두 기능은 처음부터 독립적으로 조합할 수 없었다. 원래 코드는 조합을 assert로
거부하면서 두 이유를 직접 명시했다. Unified pool 초기화에는 host pool 연결이
없었고, radix tree가 보관하는 device index는 물리 row가 아니라 compaction 중에도
안정적으로 유지되는 virtual ID였지만 HiCache transfer 경로는 이를 physical ID로
번역하지 않았다. 이 상태에서 assert만 제거하면 잘못된 row를 복사하거나, 번역
직후 compaction 또는 free/reuse가 row를 이동시켜 비동기 D2H/H2D가 다른 데이터를
읽고 쓰는 use-after-relocation 문제가 생긴다.

본 구현은 이 문제를 단계적으로 해결했다. 먼저 virtual/physical ID 계약,
stride-aware Full-KV/Mamba transfer, composite host pool, cross-pool admission 및
eviction을 구현하고 transfer마다 host-side synchronization을 수행하는 동기 기준
구현을 완성했다. 이어서 transfer 완료 CUDA event를 unified allocator에 등록하여
physical layout을 보호하는 event fencing을 추가했다. 마지막으로 L2도 KV와 Mamba가
하나의 pinned byte budget을 공유하는 typed-chunk arena로 바꾸고, Mamba D2H/H2D를
1-slot 재사용 staging buffer로 비동기화했다. Host chunk/page는 transfer event가
끝날 때까지 cross-type 재타이핑뿐 아니라 같은 KV type의 page 재사용도 금지한다.

Qwen3.5-0.8B와 Qwen3.5-4B를 사용한 L1/L2/L3 integrity, write-through,
write-back, 180회 concurrent replay 및 36회 strict probe를 통과했다. Strict
probe의 output token ID는 모두 exact 일치했고 logprob 최대 차이는 bf16
허용치 `0.1` 이하였다. 4B write-back integrity에서만 자세히 분석한
argmax near-tie 1건이 있었으며 동기 rollback에서도 재현되어 async
ordering과 분리됐다. RTX 5090 단일 GPU, CUDA Graph 비활성화 조건의
최종 성능 재검증에서 hot workload의 비동기/동기 차이는 0.8B
+2.01%, 4B -0.03%였다. L2 restore는 0.8B -0.27%, 4B -3.11%였다.
4B의 load-back 단가는 68.33 대 68.24 us/token으로 0.13% 차이였고, 비동기
실행의 cache residency와 load-back 양이 달라졌다. 따라서 event fencing은
correctness를 유지하면서 불필요한 host stall을 제거했지만, 현 실험은
안정적인 steady-state throughput 향상을 입증하지 못했다.

현재 결과는 검증한 단일 GPU, Triton backend, `page_size=1` 범위에서 실제
serving에 사용할 수 있는 수준의 correctness 근거를 제공한다. 4B L2/L3 integrity는
CUDA Graph on/off 모두 통과했다. 다만 multi-GPU/TP, 장시간 soak, 다른 L3 backend
까지 일반화한 production-ready 결론은 아직 내릴 수 없다.

## 1. 범위와 용어

이 보고서에서 다루는 질문은 “왜 `--enable-unified-memory` 자체가 없었는가”가
아니라 다음 질문이다.

> 이미 존재하던 `--enable-unified-memory`를 왜
> `--enable-hierarchical-cache`와 함께 사용할 수 없었고, 이 브랜치는 그 조합을
> 어떻게 동기 및 비동기로 구현했는가?

용어는 다음과 같다.

| 용어 | 이 보고서에서의 의미 |
|---|---|
| Unified memory | Full-KV와 Mamba/SWA state가 하나의 GPU byte buffer를 양쪽에서 동적으로 사용하는 SGLang allocator 기능 |
| L1 | Unified GPU byte buffer에 존재하는 cache state |
| L2 | HiCache의 pinned host-memory pool |
| L3 | File, Mooncake, 3FS 등 HiCache storage interface 뒤의 저장 계층 |
| Virtual ID | Radix tree와 request metadata가 보관하는 안정적인 논리 ID |
| Physical ID | 현재 unified byte buffer 안의 실제 row 위치 |
| Transfer event | D2H/H2D operation이 마지막 GPU 작업 뒤에 기록하는 CUDA event |

`--enable-unified-memory`는 CUDA의 managed allocation이나 page fault 기반 UVM과
무관하다. 이 구분을 하지 않으면 “GPU와 host 사이를 CUDA가 자동으로 migrate하면
되지 않는가?”라는 잘못된 문제 정의로 이어질 수 있다.

## 2. 배경

### 2.1 Hybrid model의 두 상태 유형

Hybrid full-attention + Mamba/GDN 모델은 성격이 다른 두 cache state를 동시에
유지한다.

1. Full-attention KV는 token 단위로 증가한다.
2. Mamba/GDN state는 request checkpoint 단위이며 convolution state와 temporal
   SSM state를 포함한다.

정적 분할에서는 Full-KV와 Mamba pool의 크기를 서버 시작 시 고정한다. 실제
workload의 prefix 길이, 동시 request 수, branch 패턴이 예상과 다르면 한쪽 pool은
남고 다른 쪽은 부족할 수 있다. Unified allocator는 하나의 byte buffer에서
Full-KV는 낮은 주소에서 위로, Mamba state는 높은 주소에서 아래로 성장시켜 이
고정 분할 손실을 줄인다.

```text
낮은 주소                                                        높은 주소
+----------+--------------------+----------------+-------------------+
| reserved | Full-KV  ----->    |  shared gap    |    <----- Mamba   |
+----------+--------------------+----------------+-------------------+
              grow-up allocator                    grow-down allocator
```

### 2.2 Virtual ID가 필요한 이유

Dynamic split은 충분한 연속 공간을 만들기 위해 live row를 compaction할 수 있다.
따라서 radix tree가 physical row를 직접 저장하면 compaction 때 tree와 모든 request
mapping을 갱신해야 한다. Unified allocator는 이 문제를 다음 indirection으로
해결한다.

```text
radix/request metadata: virtual ID
                         |
                         v
              virtual_to_physical table
                         |
                         v
shared GPU buffer의 현재 physical row
```

Compaction은 `virtual_to_physical`과 `physical_to_virtual`만 갱신한다. Tree와
request가 소유한 virtual ID는 변하지 않는다.

### 2.3 HiCache의 기존 가정

기존 HiCache transfer는 일반 KV pool의 index가 곧 device row라는 가정 아래
작성됐다. 또한 보통 layer별 KV row가 촘촘하게 배치되어 있다고 가정한다.
Unified pool에서는 두 가정이 모두 깨진다.

- Tree index는 virtual ID이며 physical row가 아니다.
- 각 layer view의 인접 token 사이 stride는 K 또는 V row 크기가 아니라 전체
  shared envelope 크기다.
- Hybrid checkpoint는 Full-KV만으로 완전하지 않고 대응하는 Mamba state도 함께
  이동해야 한다.

## 3. 기존 코드가 조합을 막은 기술적 이유

원래 `ServerArgs._handle_unified_memory_pool()`의 guard는 다음 내용을 담고
있었다.

```text
--enable-unified-memory is not yet compatible with hierarchical / host-tiered
KV cache: the unified-memory-pool init wires up no host pools, and its device
mamba / full-attention slots are VIRTUAL -- the host-offload path does not
translate them to physical.
```

이 assert는 단순한 보수적 제한이 아니었다. 다음 failure mode를 막는 correctness
barrier였다.

### 3.1 Address-space mismatch

HiCache에 virtual ID를 그대로 넘기면 transfer kernel은 그 값을 physical row로
해석한다. Virtual ID와 현재 physical ID가 우연히 같은 초기 상태에서는 문제가
숨을 수 있지만, free와 compaction이 한 번이라도 일어나면 다른 cache entry를
복사하거나 범위를 벗어난 row에 접근할 수 있다.

### 3.2 Translation-to-use race

단순히 transfer 직전에 virtual ID를 physical ID로 번역하는 것만으로는 충분하지
않다.

```text
T0  HiCache: virtual 42 -> physical 817 번역
T1  scheduler: compaction으로 virtual 42를 physical 511로 이동
T2  scheduler: physical 817을 다른 virtual ID에 재할당
T3  copy stream: 저장해 둔 physical 817에서 D2H 또는 H2D 수행
```

T3는 stale physical snapshot을 사용한다. D2H라면 다른 prefix가 L2에 저장되고,
H2D라면 새 owner의 row를 덮어쓴다. 잘못된 token이 즉시 출력될 수도 있지만,
Mamba recurrent state가 오염되면 여러 decode step 뒤에만 차이가 나타날 수도
있다.

### 3.3 Strided layout mismatch

Unified Full-KV와 Mamba layer tensor는 shared byte envelope의 view다. 일반
HiCache kernel이 contiguous row pitch를 사용하면 layer view의 실제 다음 row가
아닌 envelope 내부의 잘못된 offset으로 이동한다. 이는 ID 번역과 별개인 layout
문제다.

### 3.4 Composite state의 불완전한 이동

Hybrid prefix 재사용에는 Full-KV와 그 경계의 Mamba recurrent state가 함께
필요하다. 기존 단일 KV host pool만 구성하면 L2/L3에서 Full-KV는 hit인데 Mamba
state는 없거나 다른 checkpoint인 부분 hit를 완전한 hit로 오인할 수 있다.

### 3.5 Shared-byte admission 오류

Mamba slot 하나는 Full-KV token 하나보다 훨씬 크다. 또한 실행 중 request는
active state 외에 ping-pong tracking state를 요구할 수 있고, host hit를 복원할
destination도 필요하다. 단순 token 수 기반 admission은 shared gap을 과대평가하여
batch 구성 후 allocator가 실패하거나 peer frontier를 침범하게 한다.

### 3.6 Group allocation 수명 문제

Scheduler가 host hit를 발견하면 H2D operation은 virtual Mamba destination을
queue에 넣는다. 이 operation을 controller에 제출하기 전에 `alloc_group_end()`가
실행되면 사용하지 않은 group reservation을 반환하면서 아직 queue가 참조하는
virtual slot을 recycle할 수 있다.

### 3.7 Write policy와 L3의 composite consistency

Write-through는 node 생성 즉시, write-back은 L1 eviction 시점에 backup한다.
어느 정책이든 Full-KV anchor와 Mamba sidecar의 생명주기 및 완료 ACK가 함께
관리되어야 한다. L3 existence query도 KV file 하나만 존재한다고 hybrid entry
전체를 hit로 판정해서는 안 된다.

## 4. 설계 목표와 안전성 불변식

구현은 다음 불변식을 유지하도록 설계됐다.

| ID | 불변식 |
|---|---|
| I1 | Radix tree와 request metadata에는 virtual ID만 저장한다. |
| I2 | Physical ID는 transfer 제출 직전에 각 pool의 translator로 만든 일시적 값이다. |
| I3 | 번역된 physical row를 사용하는 transfer event가 완료되기 전에는 그 row를 relocate하거나 reuse하지 않는다. |
| I4 | H2D destination을 읽는 forward layer는 해당 layer의 producer event 이후에 실행된다. |
| I5 | Full-KV anchor와 Mamba sidecar를 하나의 논리 cache entry로 취급한다. |
| I6 | Request를 즉시 거부하고 destination을 반환하는 예외 경로는 pending load를 먼저 완료한다. |
| I7 | Mamba admission은 Full-token-equivalent shared bytes와 runtime state를 함께 계산한다. |
| I8 | L2 host chunk/page는 transfer event 전까지 재타이핑 및 same-type reuse를 금지한다. |

I1과 I2만 구현하면 address-space mismatch는 해결되지만 translation-to-use race는
남는다. 비동기 구현의 핵심은 I3과 I4를 서로 다른 event dependency로 보장한
점이다.

## 5. 구현 과정

### 5.1 1단계: Host pool과 virtual-to-physical translation

Hybrid pool assembler는 Full-KV host pool을 primary anchor로 만들고 Mamba host
pool을 `PoolName.MAMBA` sidecar로 연결한다. Unified radix component는 backup,
load-back, storage query 및 prefetch에 두 pool을 함께 전달한다.

`PoolEntry.device_index_translate_fn`은 Full-KV와 Mamba에 각각 맞는 translator를
제공한다. Queue와 tree는 virtual ID를 유지하고 controller가 실제 copy를
제출하기 직전에 physical ID를 계산한다. 번역 결과는 tree에 다시 기록하지
않는다.

주요 파일:

- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`
- `python/sglang/srt/mem_cache/hicache_storage.py`
- `python/sglang/srt/mem_cache/unified_cache/components/full_component.py`
- `python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`

### 5.2 2단계: Stride-aware transfer

Full-KV L1-to-L2 경로는 strided device row를 contiguous GPU staging buffer로
gather한 뒤 pinned host page-first pool로 복사한다. L2-to-L1은 host row를 선택한
뒤 실제 strided destination view에 `index_copy_`한다.

Mamba 경로도 runtime source/destination stride를 JIT interface에 전달한다.
L1-to-L2는 GPU staging을 사용하고, L2-to-L1은 필요한 host layer를 선택한 뒤
unified destination view에 복원한다. 큰 pinned-host Mamba row를 custom kernel이
직접 무리하게 접근할 때 발생할 수 있는 illegal memory access도 이 staging으로
피한다.

초기 호환 경로의 Mamba copy는 CPU `index_select`와 pageable `.to(cuda)`가 scheduler
thread를 동기화할 수 있었다. 현재 D2H는 unified strided row를 GPU staging으로
gather한 뒤 pinned host slot에 `non_blocking=True` DMA를 enqueue한다. H2D는 pinned
host의 layer row를 GPU staging에 올리고 device `index_copy_`로 실제 strided
destination에 scatter한다. Controller의 finish event는 같은 stream의 마지막 DMA
뒤에 기록되므로 ACK와 host-chunk release의 단일 readiness fence가 된다.

Mamba slot 하나가 수십 MiB가 될 수 있으므로 staging을 transfer batch 크기로 매번
할당하지 않는다. Component별 정확히 1 slot짜리 GPU buffer를 서버 시작 시 만들고,
여러 slot은 한 숟갈씩 순서대로 enqueue한다. Peak staging memory는 batch size와
무관하다. 현재 KV와 Mamba copy는 동일 transfer stream 안에서 순서가 정해져 있으며
서로 다른 stream으로 병렬화하지 않았다. 두 copy의 동시 실행은 PCIe bandwidth
경쟁, 두 event의 composite ACK, host chunk 수명 결합을 별도로 설계해야 한다.

주요 파일:

- `python/sglang/srt/mem_cache/pool_host/mha.py`
- `python/sglang/srt/mem_cache/memory_pool_host.py`
- `python/sglang/kernels/ops/kvcache/hicache.py`
- `python/sglang/kernels/ops/mamba/transfer_mamba.py`
- `python/sglang/kernels/jit/csrc/kvcacheio/`

### 5.3 3단계: L2 typed-chunk shared arena

기존 hybrid HiCache는 `--hicache-size`를 KV pool과 Mamba pool 각각에 적용하거나
device pool 비율대로 나누는 독립 host pool 구조였다. Unified L1의 목적과 달리 L2는
한쪽이 비어도 다른 쪽이 그 용량을 사용할 수 없었다. 새 L2는 raw pinned byte arena
하나에 겹치는 KV/Mamba tensor view를 만들고 chunk ownership metadata로 안전성을
보장한다.

```text
kv_pages_per_chunk = ceil(mamba_slot_bytes / kv_page_bytes)
chunk_bytes         = kv_pages_per_chunk * kv_page_bytes
1 Mamba slot        = 1 chunk
1 KV chunk          = kv_pages_per_chunk개의 KV page
```

Mamba state가 KV page의 정수배가 아니면 chunk 끝에 최대 KV page 하나 미만의 padding이
생긴다. 대신 Mamba slot이 1.5 chunk처럼 경계를 걸치지 않으므로 owner 변경, pointer
계산, L3 component metadata가 단순해진다. `--hicache-size N`은 unified 조합에서
KV용 N GB와 Mamba용 N GB가 아니라 두 type이 함께 쓰는 총 N GB다. Size를 지정하지
않으면 unified L1 byte budget에 `--hicache-ratio`를 곱한 값이 총 L2 budget이다.

Chunk owner는 `FREE`, `KV`, `MAMBA` 중 하나다. KV는 chunk 내부 page를 suballocate하고,
Mamba는 chunk 전체를 사용한다. 빈 chunk만 type을 바꿀 수 있다. 반대 type이 용량을
점유하면 radix metadata-aware eviction callback으로 실제 host entry를 제거한 뒤
빈 chunk를 재타이핑한다. 고정 watermark나 좌/우 영역 구분은 없다.

비동기 transfer 중에는 chunk pin count를 올린다. Pin은 다음 두 동작을 모두 막는다.

- 빈 chunk를 KV에서 Mamba 또는 반대로 재타이핑
- free된 KV page를 event 완료 전에 다른 KV entry에 재할당

두 번째 조건이 없으면 owner는 계속 KV여도 같은 byte range가 overwrite될 수 있다.
CUDA event 완료를 polling한 뒤에만 pending free/reuse를 다시 허용한다.

주요 파일:

- `python/sglang/srt/mem_cache/typed_chunk_host.py`
- `python/sglang/srt/mem_cache/pool_host/unified_chunk.py`

### 5.4 4단계: L1 shared-byte eviction과 admission

Mamba component는 `mamba_slot_full_token_cost()`를 사용해 Mamba slot 하나가
차지하는 byte를 Full-KV token 수로 환산한다. Peer Full-KV frontier가 Mamba
allocation을 막으면 Mamba checkpoint 하나만 제거하는 대신 필요한 Full-KV
token-equivalent까지 포함한 combined eviction을 요청한다.

Scheduler admission은 다음 비용을 포함하도록 보강됐다.

- request active Mamba state
- ping-pong tracking state
- host-hit restore destination
- peer frontier를 확보하기 위한 Full-KV token-equivalent bytes

이 수정은 cache가 있는 경우와 없는 경우의 batch shape가 달라지는 현상 자체를
없애려는 것이 아니라, 선택한 batch가 실제 shared allocator에 들어간다는 계약을
보장한다.

### 5.5 5단계: Scheduler queue 수명

Admitted host hit가 있으면 scheduler는 `alloc_group_end()` 전에
`ready_to_load_host_cache()`를 호출한다. 이렇게 해야 queued virtual destination이
실제 H2D operation과 allocator fence에 연결된 후 group reservation을 정리할 수
있다.

`add_one_req()`가 load를 queue한 뒤 뒤쪽 budget gate에서 request를 거부하는
경우는 정상 lifecycle과 다르다. 이 경로는 `synchronize_pending_loads()`로 pending
load를 끝낸 뒤 Mamba destination을 allocator에 반환한다.

### 5.6 6단계: Write-back과 L3

Write-back은 L1 eviction 때 필요한 entry만 L2에 기록하도록 활성화됐다. L3는
primary Full-KV page와 `.mamba` sidecar를 multi-pool API로 등록, 저장, 조회,
prefetch한다. 본 평가에서는 file backend로 non-empty KV page와 Mamba sidecar
생성을 확인했다. 다른 L3 backend는 같은 interface를 사용하지만 별도 backend
qualification이 필요하다.

## 6. 동기 기준 구현

초기 안전 구현은 unified D2H/H2D마다 finish event에
`finish_event.synchronize()`를 호출했다. Scheduler가 다시 allocator를 조작할 때는
copy가 이미 끝났으므로 stale physical row 문제가 발생하지 않는다.

이 방식은 다음 장점이 있다.

- 안전성 모델이 단순하다.
- eager/lazy compaction과 overlap scheduling을 correctness 관점에서 허용할 수
  있다.
- 문제 발생 시 transfer 완료 시점이 명확해 디버깅이 쉽다.

반면 CPU thread가 transfer 완료까지 막히므로 그 시간 동안 다음 scheduler work를
준비할 수 없다. 여기서 “동기”는 CUDA copy 자체를 synchronous API로 바꿨다는
의미보다, copy stream에 기록한 event를 host가 기다린다는 의미다.

동기 동작은 비동기 구현 이후에도 rollback 및 A/B 기준선으로 남아 있다.

```bash
export SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS=1
```

## 7. 비동기 event-fenced 구현

### 7.1 전체 protocol

기본값에서는 controller가 host synchronization 대신 finish event를 allocator에
등록한다.

```text
HiCache controller                Unified allocator              Forward
       |                                  |                         |
       | translate virtual -> physical    |                         |
       | enqueue D2H/H2D                   |                         |
       | record finish_event               |                         |
       | register_external_transfer_event  |                         |
       |---------------------------------->|                         |
       | return to scheduler               |                         |
       |                                  | fresh allocation 가능    |
       |                                  | unrelated free 가능      |
       |                                  |                         |
       |                                  | relocation/reuse 필요 시 |
       |                                  | stream.wait_event         |
       |                                  |                         |
       | per-layer H2D producer event ----------------------------->|
       |                                  |               layer read 전 wait
```

Host thread는 정상 경로에서 `Event.synchronize()`를 호출하지 않는다. GPU stream
사이의 dependency만 추가하므로 CPU scheduler는 다음 작업을 계속할 수 있다.

### 7.2 Allocator의 hazard 처리

`MultiEndedAllocator.register_external_transfer_event()`는 아직 완료되지 않은
event를 보관한다. 이 구현은 row별 pin count 대신 현재 physical layout 전체에 대한
보수적 fence를 사용한다.

Allocator operation별 정책은 다음과 같다.

| Operation | Pending transfer가 있을 때의 동작 |
|---|---|
| Fresh frontier allocation | 기존 row를 움직이지 않으므로 계속 진행 |
| Unrelated virtual free | physical row를 즉시 덮어쓰지 않으면 계속 진행 |
| Non-urgent lazy compaction | 이번 tick의 compaction을 미루고 반환 |
| Urgent compaction | scheduler CUDA stream에 `wait_event`를 넣은 뒤 relocation |
| Eager free/relocation | scheduler CUDA stream에 `wait_event`를 넣은 뒤 이동 |
| Freed-hole reuse | hole을 새 virtual ID에 bind하기 전에 `wait_event` 삽입 |

Composite Mamba/SWA allocator는 같은 event를 양쪽 sub-allocator에 전달한다.
Full-KV와 sidecar 중 어느 쪽 physical layout도 transfer 도중 바뀌지 않는다.

### 7.3 H2D producer-consumer ordering

Allocator fence는 “destination row가 이동하거나 재사용되지 않는다”만 보장한다.
Forward가 copy보다 먼저 그 row를 읽지 않는다는 보장은 별도로 필요하다. 기존
HiCache `LayerDoneCounter`와 per-layer load event가 이 producer-consumer ordering을
제공한다. 각 forward layer는 자신이 소비할 H2D layer event 이후에 실행된다.

즉 두 event의 역할은 다르다.

| Event dependency | 막는 race |
|---|---|
| Transfer event -> allocator relocation/reuse | copy가 stale physical row를 사용하는 문제 |
| Per-layer producer event -> forward layer | forward가 H2D 완료 전 destination을 읽는 문제 |

### 7.4 왜 완전한 무대기가 아닌가

비동기 구현은 host blocking을 제거하지만 모든 dependency를 제거하지 않는다.
다음 경우에는 H2D가 여전히 request의 critical path다.

- 복원할 layer가 준비되어야 그 layer forward를 진행할 수 있다.
- Shared gap이 부족하여 allocator가 즉시 relocation해야 한다.
- Lazy free가 만든 hole을 곧바로 재사용해야 한다.
- Cache restore가 recomputation보다 느린 짧은 synthetic prefix다.

따라서 “asynchronous”는 correctness dependency를 없앤다는 뜻이 아니라,
dependency를 host-wide synchronization에서 필요한 CUDA stream ordering으로
좁힌다는 뜻이다.

## 8. Write-through와 hot-cache workload의 해석

### 8.1 Write-through를 기본 운영에 보수적으로 권장하는 이유

Write-through는 새 cache node를 L1에 만들 때마다 L2 backup도 시작한다. 이후 한
번도 재사용되지 않을 prefix까지 전송하므로 다음 비용이 발생한다.

- D2H bandwidth와 copy stream 사용
- GPU staging 및 pinned host-memory traffic
- controller queue, ACK 및 radix metadata 갱신
- forward kernel과의 메모리 대역폭/stream 경쟁
- L2/L3까지 이어질 때 write amplification과 storage metadata 비용

Write-back은 L1에서 실제로 밀려나는 entry만 backup하므로 cold 또는 재사용률이
낮은 workload에서 불필요한 쓰기를 줄인다. 다만 eviction 순간에 backup latency가
몰릴 수 있고, 아직 write-back되지 않은 L1 entry는 하위 tier에서 즉시 재사용할 수
없다는 trade-off가 있다. 본 구현은 두 정책의 correctness를 모두 검증했지만,
운영 기본값은 하위 tier의 즉시 가용성 요구와 prefix reuse 분포를 측정한 뒤
선택해야 한다.

### 8.2 Hot-cache workload란 무엇인가

본 benchmark의 `hot` scenario는 population 이후 working set이 L1 GPU cache 안에
남아 반복 요청에서 L2 load-back이 0인 workload다. 측정된 cached-token ratio는
약 96.3%였다. 이 workload는 HiCache가 데이터를 실제로 복원하는 성능이 아니라
다음 feature overhead를 측정한다.

- HiCache controller와 radix metadata 관리
- scheduler tick의 ACK/event polling
- cache policy에 따른 batch와 bookkeeping 변화
- server launch별 allocator/cache 초기화 차이

Hot 결과에서 sync와 async 차이가 작다면 transfer synchronization이 아니라 이
공통 control-plane 비용과 run-to-run variance가 주된 요소라는 뜻이다.

## 9. 평가 방법

### 9.1 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| 모델 | `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B` |
| Parallelism | TP=1, DP=1, PP=1, 단일 GPU |
| Attention/linear/Mamba backend | Triton |
| HiCache I/O | `kernel` |
| Page size | `1` |
| L3 correctness backend | File |
| Scheduling | Overlap scheduling enabled |
| Mamba strategy | `extra_buffer` |
| CUDA Graph | 성능/스트레스는 off, 4B L2/L3 integrity는 off와 default graph-on |

대화 중 편의상 “Qwen3.5-0.9B”라고 부른 모델의 실제 Hugging Face model ID와
실험 대상은 `Qwen/Qwen3.5-0.8B`다.

### 9.2 Correctness oracle

서버가 시작되거나 한 번 generation에 성공했다는 사실만으로 cache correctness를
판정하지 않았다. Integrity test는 다음을 모두 요구한다.

1. 60개(4B는 80개) 서로 다른 긴 prompt로 L1과 L2 capacity를 초과한다.
2. `evicted_tokens_total` 증가로 L1 eviction을 증명한다.
3. `backuped_tokens_total`, non-empty L3 file 및 `.mamba` sidecar로 composite L3
   backup을 증명한다.
4. Counter 변화로 L2-only hit와 L3 hit를 동적으로 찾는다.
5. 최초 계산과 restore 후 output token ID를 우선 exact 비교한다.
6. Logprob가 finite인지, logprob token ID가 같은지 확인한다.
7. 비교 가능한 동일 token prefix의 bf16 logprob 절대 차이를 `0.1` 이하로 제한한다.
8. 첫 불일치가 있으면 양쪽 선택 token이 서로의 top-5에 있고, 두 실행 모두에서
   두 token의 logprob gap이 `0.15` 이하인 argmax near-tie만 별도 분류한다.
9. Restore 후 `/health`가 HTTP 200인지 확인한다.

Serving의 near-tie 허용은 transfer byte correctness를 대신하지 않는다. 별도 단위
테스트는 KV와 Mamba flat-page serialization을 non-uniform byte pattern으로 왕복해
bit-exact 비교하고, file backend가 쓰는 raw pointer/size metadata가 정확한 chunk와
component를 가리키는지 검증한다. CUDA round-trip은 fp16/bf16에서 D2H/H2D 결과를
원본 tensor와 비교한다.

Stress test는 60개 baseline prompt에 대해 8 client worker가 순서를 번갈아 가며
3 round, 총 180회 concurrent replay한다. 매 round 후 batch-shape 차이를 제거한
12개 sequential strict probe를 수행하여 총 36개 probe의 output ID와 logprob를
baseline과 비교한다.

Concurrent phase의 greedy token은 bf16 near-tie에서 batch shape만 달라도 바뀔 수
있다. 따라서 concurrent response는 내부 token/logprob consistency와 finite 값을
검사하고, baseline과 다른 token 수를 관측치로 기록한다. Cache corruption의 strict
oracle은 각 churn round 뒤 동일한 sequential shape로 수행한 probe다.

### 9.3 Performance protocol

`benchmark/hicache/bench_unified_hicache_async.py`는 매 mode를 독립 server launch로
실행하고 JSON artifact를 남긴다. 비교 mode는 다음과 같다.

| Mode | 설정 | 목적 |
|---|---|---|
| Unified only | Unified memory, HiCache 없음 | Cache hierarchy를 추가하지 않은 기준선 |
| Sync HiCache | Unified + HiCache + `SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS=1` | 동기 구현 기준선 |
| Async HiCache | Unified + HiCache, env 기본값 `0` | Event-fenced 구현 |

`hot`은 12 prompt, 8 worker, 4 round 중 앞 1 round를 버린 독립 launch
1회/mode의 smoke 측정이다. `restore`는 0.8B에서 60 prompt, 8 worker,
3 round 중 앞 1 round를 버린 launch 2회/mode를 사용했다. 4B는 메모리
제약에 맞춰 30 prompt, 4 worker, 같은 3 round 구성을 사용했고 Unified-only
2회, sync/async 각 3회를 서로 다른 순서로 기동했다. L2는 0.8B
1 GB, 4B 2 GB였다. 모든 성능 실험은 write-back과 graph-off를 사용했다.

## 10. Correctness 결과

### 10.1 L2/L3 integrity

| 모델 | 정책 | L2 token / 차이 | L3 token / 차이 | Output 판정 | 결과 |
|---|---:|---:|---:|---|---|
| Qwen3.5-0.8B | write-through | 2,751 / 0.028611 | 2,559 / 0.052683 | exact | PASS |
| Qwen3.5-0.8B | write-back | 2,751 / 0.028611 | 2,559 / 0.052683 | exact | PASS |
| Qwen3.5-4B | write-through | 2,047 / 0.000000 | 2,752 / 0.073050 | exact | PASS |
| Qwen3.5-4B | write-back | 2,816 / 0.019538 | 2,432 / 0.048886 | L2 exact, L3 near-tie 1 | PASS |

4B write-back L3의 첫 불일치는 decode position 6이었다. 최초 실행은 token 24가
token 23보다 0.125 높았고 restore 실행은 두 token이 정확히 동률이었다. 동기
rollback에서도 같은 불일치가 반복되어 async ordering과 분리했으며, 최종 async
artifact는 이를 near-tie로 기록했다. 비교 가능한 앞 6 token은 exact였고 최대
logprob 차이는 0.048886이었다. 나머지 row는 output token ID와 logprob token ID가
exact 일치했다. 모든 run의 health는 200이고 non-empty Full-KV file과 Mamba
sidecar를 생성했다.

### 10.2 Concurrent tier-churn stress

| 모델 | Concurrent replay | Strict probe | L1 evict | L2 load-back | L3 prefetch | 최대 logprob 차이 | Concurrent mismatch | 결과 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-0.8B | 180 | 36 | 673,882 | 144,294 | 125,190 | 0.089145 | 3 | PASS |
| Qwen3.5-4B | 180 | 36 | 667,054 | 66,399 | 64,357 | 0.078532 | 1 | PASS |

Concurrent mismatch는 strict probe 실패가 아니다. 서로 다른 batch
shape에서 발생한 greedy near-tie 차이이며, 각 response 내부의 output/logprob token
ID는 일치했다. 두 모델 모두 churn 이후 동일한 sequential 조건으로 다시 실행한
36개 probe는 전부 exact token match를 통과했고 near-tie 분류도 0건이었다.

4B stress artifact는 typed-chunk/async-Mamba staging 구현에서 생성했지만,
이후 추가한 “전송 중 같은 KV type page 재사용 금지” guard보다는 앞선다.
해당 guard 이후에는 실제 CUDA event reuse test, 전체 `169 passed, 1 skipped`
회귀 묶음 및 4B L2/L3 integrity를 다시 통과했지만 4B 180-replay stress
전체는 재실행하지 않았다. Guard는 재사용을 더 보수적으로 막는 변경이지만,
이 시점 차이는 다른 GPU의 최종 TP 행렬에서 재확인해야 한다.

### 10.3 단위 및 race test

`run_alltest.sh --skip-serving`로 allocator, unified dispatch, staged transfer,
admission, Mamba transfer 및 CUDA transfer 10개 invocation을 fail-fast 실행한 결과는
`169 passed, 1 skipped`였고 실패는 없었다. 1 skip은 해당 fixture에서 지원하지 않는
Mamba staged-dispatch variant다. TP=1 serving은 별도 artifact 시험으로 수행했고,
TP=2/4는 현재 visible GPU가 1개라 실행하지 않았다.

특히 다음 CUDA race를 인위적으로 만들었다.

- `torch.cuda._sleep`으로 D2H reader를 늦춘 뒤 eager relocation 수행
- Reader가 끝나기 전에 lazy free가 만든 hole을 새 allocation이 재사용하도록 압박
- Reader가 overwrite된 새 값이 아니라 transfer 시작 시점의 원래 row를 읽는지
  확인

CPU mock test는 non-urgent compaction defer, urgent stream wait, 정상 경로의 host
`synchronize()` 부재, rejected-load drain, fragmented cross-type reclaim 및
same-type pinned-page reuse 금지를 각각 검증했다. CUDA test는 실제 event 완료 전
KV page가 재할당되지 않는지도 확인했다. Python compile과 `git diff --check`도
통과했다.

이 수치는 저장소 전체 test suite가 아니라 본 변경과 직접 관련된 선택 test
suite의 결과다. 4-GPU pipeline-parallel test는 unified-memory 조합을 사용하지 않고
현재 장비 범위를 벗어나므로 본 결론의 근거에 포함하지 않았다.

## 11. 성능 결과

### 11.1 Input throughput

단위는 input token/s이며 restore의 `평균 ± 독립 launch 간
표준편차`다. Hot은 mode당 launch 1회이므로 표준편차를 표시하지 않았다.

| 모델/시나리오 | Unified only | Sync HiCache | Async HiCache | Async 대 Sync |
|---|---:|---:|---:|---:|
| 0.8B hot | 31,025.4 | 33,048.3 | 33,713.9 | +2.01% |
| 0.8B restore | 32,395.0 ± 436.8 | 19,406.5 ± 200.2 | 19,354.7 ± 9.9 | -0.27% |
| 4B hot | 23,959.1 | 23,887.2 | 23,880.7 | -0.03% |
| 4B restore | 26,266.2 ± 67.8 | 18,157.3 ± 635.7 | 17,593.4 ± 583.9 | -3.11% |

4B sync/async의 세 번째 인접 실행쌍은 18,405.0과 18,185.3 token/s로
차이가 -1.19%였다. 세 쌍의 개별 차이는 -5.66%, -2.39%, -1.19%로
독립 server launch 편차가 작은 차이보다 컸을 수 있음을 보여준다.

### 11.2 TTFT

단위는 초다.

| 모델/시나리오 | Unified only | Sync HiCache | Async HiCache |
|---|---:|---:|---:|
| 0.8B hot | 0.0474 | 0.0449 | 0.0428 |
| 0.8B restore | 0.6534 | 1.0840 | 1.0883 |
| 4B hot | 0.0603 | 0.0587 | 0.0598 |
| 4B restore | 0.3969 | 0.5820 | 0.5968 |

### 11.3 결과 해석

첫째, hot workload에서 HiCache 및 async overhead는 측정 noise에 가까웠다.
실제 load-back이 0이므로 event fencing의 data-path 이점보다 controller/metadata
비용과 launch variance를 본 결과다.

둘째, 실제 restore에서 async는 sync보다 안정적으로 빠르지 않았다. 0.8B의
load-back 단가는 sync 52.37, async 51.28 us/token으로 async가 약간
낮았지만 throughput은 0.27% 낮아 실질적으로 같았다. 4B는 sync 68.24,
async 68.33 us/token으로 전송 단가가 0.13% 차이였다. 비동기의 평균
load-back은 45.4k token, sync는 44.0k token이었고 cache ratio와 batch 구성도
launch별로 달랐다. 즉 4B의 3.11% 차이를 memcpy 자체의 속도 저하로
해석할 근거는 없다.

셋째, event fencing이 제거한 것은 host가 모든 transfer 완료를 기다리는
구간이다. 다음 비용은 그대로 남는다.

- forward가 필요한 layer의 H2D 완료를 기다리는 critical path
- allocator pressure 시 relocation/reuse 전에 들어가는 stream-side wait
- Full-KV와 큰 Mamba recurrent state의 실제 PCIe transfer
- HiCache controller, radix metadata, ACK/event polling
- Cache state 차이로 인한 batch 구성 및 scheduling 변화

넷째, 이 synthetic restore에서는 HiCache가 recomputation보다 느렸다. 0.8B는
Unified-only 대비 약 40%, 4B는 약 31~33% 낮은 input throughput을 보였다. 이는
HiCache가 일반적으로 느리다는 결론이 아니다. Output을 1 token으로 제한한 비교적
짧은 prefix, 단일 GPU, PCIe/host 특성에서 restore 비용이 saved prefill compute보다
컸다는 뜻이다. 더 긴 공유 prefix, 계산량이 큰 모델, 반복 reuse, 다른 interconnect
또는 L3 특성에서는 손익분기점이 달라진다.

### 11.4 성능 결론

비동기 구현은 동기 구현의 불필요한 host blocking을 구조적으로 제거했다. 그러나
현재 측정으로 주장할 수 있는 결론은 다음까지다.

- Hot workload에서 추가 regression은 관측되지 않았다.
- 0.8B restore에서 async와 sync는 실질적으로 같았다.
- 4B restore의 async는 평균 3.11% 낮았지만 인접 쌍의 차이는
  1.19%였고, load-back 단가는 동일했다.
- 독립 launch 수와 측정 시간이 적어 작은 차이는 통계적으로 확정할 수 없다.
- “비동기화로 steady-state throughput이 향상됐다”는 주장은 현재 근거로는
  성립하지 않는다.

## 12. CUDA Graph와 correctness 범위

### 12.1 왜 성능 및 stress 평가에서 CUDA Graph를 껐는가

주 성능 비교와 concurrent stress는 decode와 prefill CUDA Graph를 모두
비활성화했다. 목적은 다음과 같다.

1. Unified allocator/HiCache event ordering만 독립적으로 검증한다.
2. Captured static buffer와 dynamic host-load metadata의 상호작용을 별도 변수로
   분리한다.
3. 동일 eager execution path에서 sync와 async의 차이만 비교한다.
4. Graph capture/warm-up 시간과 bucket 선택이 짧은 benchmark를 왜곡하지 않게 한다.

### 12.2 CUDA Graph-on 추가 integrity 결과

원칙적으로 결과가 달라질 수 있다. CUDA Graph는 pointer, buffer 및 일부 실행 metadata를 capture 시점의
형태로 재사용한다. Unified allocator는 physical mapping을 동적으로 바꾸고,
HiCache H2D는 request마다 다른 producer event를 만든다. Graph replay 전에 다음을
보장하지 못하면 stale mapping 또는 load-before-read가 생길 수 있다.

- Replay가 현재 virtual-to-physical mapping을 다시 반영한다.
- Dynamic H2D producer event wait가 graph 밖이나 적절한 graph break에 삽입된다.
- Graph가 읽는 shared metadata를 allocator relocation이 덮어쓰기 전에 WAR
  ordering을 건다.

본 변경 후 Qwen3.5-4B, TP=1, default CUDA Graph 설정으로 write-through와
write-back L2/L3 integrity를 각각 실행했다. 두 정책 모두 L1 eviction, L2-only
load-back, L3 prefetch, KV file/Mamba sidecar, exact output ID 및 health 200을 통과했다.
Graph capture는 token bucket 42개와 batch bucket 4개를 실제 생성했다.

| 정책 | L2 token | L2 최대 logprob 차이 | L3 token | L3 최대 logprob 차이 | Output ID |
|---|---:|---:|---:|---:|---|
| write-through | 2,047 | 0.000000 | 2,752 | 0.081721 | exact |
| write-back | 2,816 | 0.019538 | 2,879 | 0.057164 | exact |

이는 “Graph를 켜면 현재 확인된 4B integrity가 깨진다”는 가설을 반박한다. 그러나
Graph-on concurrent churn, TP=2/4, 다른 graph backend 조합까지 보증하지는 않는다.

현재 server-argument validation은 unified memory에서 TC piecewise prefill graph를
거부하지만 monolithic decode graph 등 모든 허용 조합의 correctness를 보증한다는
뜻은 아니다. CLI에서 시작할 수 있다는 사실과 production qualification은 구분해야
한다.

### 12.3 현재 운영 권고

본 보고서의 검증 범위를 그대로 재현하려면 다음 중 하나로 CUDA Graph를 끈다.

```bash
--disable-cuda-graph
```

또는 phase별로 명시한다.

```bash
--cuda-graph-backend-decode disabled \
--cuda-graph-backend-prefill disabled
```

Graph-on을 일반 운영 범위로 넓히기 전에 다음 matrix가 더 필요하다.

- Decode `full`, prefill `breakable`
- L2-only와 L3 restore (4B TP=1 완료)
- Write-through와 write-back (4B TP=1 완료)
- Eager와 lazy compaction
- Restore와 decode가 겹치는 concurrent churn
- 0.8B graph-on 및 4B TP=2/4 token/logprob oracle
- Nsight Systems에서 H2D event wait와 graph replay ordering 확인

장기적으로는 HiCache loading이 있는 prefill batch만 eager로 보내고 decode graph는
유지하거나, dynamic per-layer wait를 breakable graph segment 밖에 삽입하는 방식이
보수적인 해법이다.

## 13. 지원 범위와 남은 제한

### 13.1 구현 및 GPU 검증 범위

| 항목 | 검증 값 |
|---|---|
| Model family | Hybrid full-attention + Mamba/GDN, Qwen3.5 |
| I/O backend | `kernel` |
| Page size | `1` |
| Write policy | `write_through`, `write_back` |
| Scheduling | Overlap enabled |
| Compaction | Eager/lazy 관련 race test 포함 |
| L3 | File backend end-to-end |
| Backend | Triton attention, linear attention, Mamba |
| GPU topology | NVIDIA CUDA, single GPU |
| CUDA Graph | Off 전 범위, default graph-on 4B TP=1 integrity |

### 13.2 현재 미지원 또는 미검증

- Unified memory + LMCache
- PD disaggregation
- Speculative decoding
- Decode context parallelism(`dcp_size > 1`)
- Direct HiCache I/O
- `page_size > 1`
- TC piecewise prefill CUDA Graph
- TP/PP multi-GPU correctness와 성능
- File 이외 L3 backend의 unified Mamba sidecar end-to-end 검증
- 장시간 soak, fault injection, process crash recovery
- CUDA Graph-on concurrent stress와 TP=2/4

Direct I/O는 contiguous device row를 가정하므로 unified strided view와 맞지 않는다.
`page_size > 1`은 Full-KV page와 Mamba state slot의 physical addressing unit을
allocator 수준에서 통합해야 하므로 단순 transfer 수정으로 해결할 수 없다.

## 14. 실제 serving 사용성 판단

현재 구현은 다음 조건에서 실험적 serving에 사용할 만하다.

- Qwen3.5 hybrid model
- 단일 NVIDIA GPU
- Triton attention/linear/Mamba backend
- HiCache kernel I/O, `page_size=1`
- CUDA Graph off 또는 검증한 4B TP=1 default graph 조합
- Write-through 또는 write-back
- 배포 전 실제 prefix 길이와 reuse 분포로 별도 benchmark 수행

Correctness 근거는 단일 smoke test보다 강하다. L1/L2/L3 tier movement, output
token ID, logprob, Mamba sidecar, concurrent churn 및 allocator race를 함께 확인했다.
동시에 다음 이유로 “일반 production ready”라고 부르기는 이르다.

- CUDA Graph-on stress와 multi-GPU 조합은 아직 미검증이다.
- 단일 GPU와 두 모델 크기만 평가했다.
- Async가 sync 대비 일관된 throughput 향상을 보이지 않았다.
- Restore가 recomputation보다 이득인 workload 영역을 아직 체계적으로 찾지 않았다.
- 장시간 serving과 backend 장애 복구를 검증하지 않았다.

안전한 rollout은 graph-off canary에서 시작해 cache-hit source, load-back duration,
TTFT, retraction, health 및 server log를 수집하고, workload별로 unified-only와
비교하는 방식이 적절하다. 문제가 생기면 코드 rollback 없이 다음 env로 transfer만
동기 기준선으로 되돌릴 수 있다.

```bash
export SGLANG_HICACHE_SYNC_UNIFIED_TRANSFERS=1
```

## 15. 재현 방법

### 15.1 기본 서버

다음은 본 평가 범위에 맞춘 단일 NVIDIA GPU, graph-off 예시다.

```bash
PYTHONPATH="$PWD/python" \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-0.8B \
  --enable-unified-memory \
  --enable-hierarchical-cache \
  --hicache-size 1 \
  --hicache-write-policy write_back \
  --hicache-io-backend kernel \
  --page-size 1 \
  --attention-backend triton \
  --linear-attn-backend triton \
  --mamba-backend triton \
  --mamba-radix-cache-strategy extra_buffer \
  --disable-cuda-graph \
  --enable-metrics
```

File-backed L3를 추가하려면 storage directory와 backend를 지정한다.

```bash
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/path/to/hicache

# 위 명령에 추가
--hicache-storage-backend file
```

### 15.2 Correctness test

0.8B write-through, write-back integrity:

```bash
PYTHONPATH=python \
python test/registered/hicache/test_unified_memory_hicache_integrity.py
```

0.8B concurrent overlap stress:

```bash
PYTHONPATH=python \
python test/registered/hicache/test_unified_memory_hicache_overlap_stress.py
```

4B integrity:

```bash
SGLANG_UNIFIED_HICACHE_TEST_MODEL=Qwen/Qwen3.5-4B \
SGLANG_UNIFIED_HICACHE_PRESSURE_REQUESTS=80 \
SGLANG_UNIFIED_HICACHE_MEM_FRACTION_STATIC=0.40 \
PYTHONPATH=python \
python test/registered/hicache/test_unified_memory_hicache_integrity.py
```

4B stress:

```bash
SGLANG_UNIFIED_HICACHE_TEST_MODEL=Qwen/Qwen3.5-4B \
SGLANG_UNIFIED_HICACHE_MEM_FRACTION_STATIC=0.40 \
PYTHONPATH=python \
python test/registered/hicache/test_unified_memory_hicache_overlap_stress.py
```

결과 artifact를 남기려면 다음 환경 변수를 추가한다.

```bash
export SGLANG_UNIFIED_HICACHE_TEST_ARTIFACT_DIR=/path/to/test-results
```

각 run은 `result.json`, `environment.json`, `server.log`, phase별 Prometheus
snapshot과 integrity test의 `storage_manifest.json`을 남긴다. L3 payload 자체는
크므로 test 종료 때 제거한다.

### 15.3 세 mode 성능 비교

```bash
PYTHONPATH=python python benchmark/hicache/bench_unified_hicache_async.py \
  --mode unified --scenario restore --launch-id restore-unified-1

PYTHONPATH=python python benchmark/hicache/bench_unified_hicache_async.py \
  --mode sync --scenario restore --launch-id restore-sync-1

PYTHONPATH=python python benchmark/hicache/bench_unified_hicache_async.py \
  --mode async --scenario restore --launch-id restore-async-1
```

정식 성능 비교에서는 각 mode를 최소 3회 독립 launch하고 mode 순서를
회전하는 것을 권장한다. 본 최종 실험은 위 9.3절에 적은 것처럼 hot
1회/mode, restore 2~3회/mode이므로 작은 차이를 확정적 결론으로 취급하지
않았다. 같은 server process에서 mode를 바꾼 결과는 allocator 초기 상태와
CUDA warm-up이 공유되므로 독립 비교로 취급하지 않는다.

## 16. 코드와 커밋 지도

### 16.1 주요 코드

| 영역 | 파일 |
|---|---|
| CLI compatibility | `python/sglang/srt/server_args.py` |
| Shared byte buffer와 unified pool | `python/sglang/srt/mem_cache/unified_memory_pool.py` |
| Virtual/physical allocator와 event fence | `python/sglang/srt/mem_cache/multi_ended_allocator.py` |
| Unified radix와 load lifecycle | `python/sglang/srt/mem_cache/unified_radix_cache.py` |
| Scheduler admission 및 queue flush | `python/sglang/srt/managers/scheduler.py` |
| Hybrid transfer controller | `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` |
| Host-pool assembly | `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` |
| Shared L2 typed-chunk allocator | `python/sglang/srt/mem_cache/typed_chunk_host.py` |
| Unified KV/Mamba host adapters | `python/sglang/srt/mem_cache/pool_host/unified_chunk.py` |
| Full-KV host transfer | `python/sglang/srt/mem_cache/pool_host/mha.py` |
| Mamba host transfer | `python/sglang/srt/mem_cache/memory_pool_host.py` |
| Rollback env | `python/sglang/srt/environ.py` |
| End-to-end tests | `test/registered/hicache/test_unified_memory_hicache_*.py` |
| CUDA race tests | `test/registered/kernels/ops/kvcache/test_unified_hicache_transfer.py` |
| Shared L2/race tests | `test/registered/{unit,kernels}/**/test_*typed_chunk*`, `test_unified_chunk_hicache_transfer.py` |
| Benchmark | `benchmark/hicache/bench_unified_hicache_async.py` |

### 16.2 구현 진화

| Commit | 역할 |
|---|---|
| `4a8e76805` | Hybrid model용 unified memory pool 도입 |
| `d6a1585be` | Unified HiCache L1/L2 write-through 및 physical translation 시작 |
| `fd097666f` | Host restore 경로 수정 |
| `3b9c65fc8` | Transfer 동기화를 이용해 overlap scheduling/compaction 허용 |
| `450c5280b` | Write-back 지원 |
| `636373f7d` | Composite L3 storage 지원 |
| `749738e71`~`b9aab1ec2` | Tier integrity, artifact 및 concurrent churn test 강화 |
| `271ed89b7` | Admission, group lifetime, sync transfer correctness 강화 및 CUDA race test 추가 |
| `6458f2e70` | Host synchronization을 allocator-aware CUDA event fencing으로 교체 |
| `99b384342` | Shared L2 typed chunks, async Mamba staging, host-chunk event pinning 추가 |

구현을 review할 때는 위 순서대로 `git show <commit>`을 확인하면 기능 의존성과
각 guard가 제거된 이유를 가장 쉽게 추적할 수 있다.

## 17. 위협 요인과 후속 연구

본 결과에는 다음 validity limitation이 있다.

- GPU 한 종류와 단일 node만 사용했다.
- Hot은 mode당 1회, restore는 2~3회 launch이고 측정 시간이 짧다.
- Metrics와 info logging이 켜져 있어 절대 성능에 관측 비용이 포함된다.
- 주 성능/stress는 CUDA Graph를 껐고 graph-on은 4B integrity만 포함한다.
- Prefix 길이와 cache ratio가 제한된 synthetic workload다.
- Nsight Systems 기반 stream timeline과 scheduler idle-time 분해를 완료하지 않았다.
- File L3의 correctness는 확인했지만 원격 L3의 network/storage variance는 포함하지
  않았다.

후속 작업의 우선순위는 다음과 같다.

1. Graph-on concurrent churn 및 TP=2/4 correctness matrix
2. HiCache-loading prefill batch의 graph backend별 ordering audit
3. 30분 이상 soak 및 request cancellation/failure injection
4. TP=2 이상 multi-GPU validation
5. Nsight Systems로 H2D, forward, allocator wait와 idle gap 분해
6. Prefix length/reuse count를 변화시켜 restore와 recomputation의 손익분기점 측정
7. Global event fence를 실제 row-range pinning으로 세분화할 때의 비용/효과 평가

## 18. 결론

기존 assert는 제거하기 불편한 임시 제한이 아니라 실제 데이터 오염 가능성을 막는
안전장치였다. Unified allocator의 virtual ID, 동적 compaction, strided layout과
Mamba composite state는 기존 HiCache의 physical contiguous KV-row 가정과 직접
충돌했다.

이 브랜치는 host pool 조립, per-pool ID translation, stride-aware transfer,
shared-byte admission, scheduler lifetime ordering 및 composite L3를 구현해 기능적
간극을 메웠다. 동기 구현으로 먼저 correctness boundary를 세운 뒤, transfer finish
event를 allocator의 relocation/reuse dependency로 바꾸어 정상 scheduler 경로의
host blocking을 제거했다. Qwen3.5-0.8B와 4B의 tier restore 및 concurrent churn은
검증 범위에서 correctness를 지지한다.

성능 결론은 더 제한적이다. Async event fencing은 설계상 필요한 최적화지만 현재
workload에서는 sync 대비 안정적인 throughput 개선을 보이지 않았다. 실제 H2D와
forward dependency가 critical path에 남기 때문이다. 따라서 현 구현은 graph-off
단일 GPU canary에는 사용할 만하지만, CUDA Graph-on과 multi-GPU를 포함한 일반
production 배포 전에는 추가 qualification이 필요하다.
