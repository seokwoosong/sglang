# Unified Mamba deferred-free 재현 실험 메모

## 현재 결론

- CPU 회귀 테스트에서는 버그와 패치 효과가 결정적으로 재현된다.
- RTX 5090에서 Qwen3.5-0.8B SGLang 서버는 정상 구동된다.
- 실제 서버에서는 아직 `Can not alloc mamba cache` assert가 발생하지 않았다.
- 다만 assert에 필요한 두 현상은 로그에서 각각 확인했다.
  1. Mamba 슬롯 부족으로 Full KV를 donor로 축출하는 경로
  2. 열린 `free_group`에 Full KV 반환이 보류되는 현상
- 아직 두 현상이 **동일한 Mamba retry 순간**에 겹치지 않았다.

## 버그 원리

Prefill 결과 처리는 batch 전체에 대해 `free_group`을 연다. 이 안에서
unfinished request의 Mamba checkpoint 슬롯 할당이 실패하면 Full KV를
축출해 공간을 빌린다.

패치 전에는 Full KV 반환이 그룹 끝까지 보류되므로 즉시 실행되는 Mamba
retry가 그 공간을 보지 못할 수 있다. 패치는 retry 전에
`flush_deferred_frees()`를 호출해, 바깥 `free_group`은 유지하면서 보류된
Full 반환만 즉시 반영한다.

## Qwen3.5-0.8B가 적합한 이유

- Full-attention layer: 6
- Gated DeltaNet layer: 18
- Full KV 한 토큰: 12,288 bytes
- Mamba state 한 슬롯: 19,537,920 bytes
- Mamba 한 슬롯과 같은 크기:

```text
ceil(19,537,920 / 12,288) = 1,590 Full tokens
```

CPU reproducer가 사용하는 `1590`과 정확히 같다.

## 비교 대상

```text
37fdc1399  pre-patch CPU reproducer
f600c3dff  flush_deferred_frees 패치
```

회귀 테스트:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/mem_cache/test_multi_ended_allocator.py \
  -k 'DeferredFullFree or full_tokens_for_mamba_slots'
```

패치 브랜치 결과: `3 passed`.

## 실제 서버에서 확인한 것

작은 통합 풀 설정:

```text
--max-total-tokens 8192
--max-mamba-cache-size 24
--max-running-requests 8
--page-size 1
--mamba-radix-cache-strategy no_buffer
--disable-overlap-schedule
```

RTX 5090에서 관측된 풀:

```text
total_bytes = 569,573,376 bytes
Full max_slots = 46,352
Mamba max_slots = 29
Mamba 1 slot ~= Full 1,590 tokens
```

관측 결과:

- 7K 입력 여러 개로 Full KV를 채울 수 있었다.
- Mamba miss가 발생해 Full 7,089 tokens와 Mamba 3 slots를 축출한 로그를
  확인했다. 이때는 `free_group` 밖이었다.
- 다른 batch에서는 `free_group` 종료 시 Full 3,577 tokens가 보류돼 있던
  로그를 확인했다. 이때는 Mamba retry가 없었다.
- 마지막 8개 동시 요청은 실제 scheduler에서 `1 + 7` batch로 분리됐다.
- 그 전에 Full 캐시가 약 21K에서 7K로 축출되어 Mamba 여유가 생겼고,
  assert 없이 8개 요청이 모두 HTTP 200으로 끝났다.
- 24K 단일 입력 방식은 `max-total-tokens=8192` 요청 한도 때문에 HTTP
  400으로 거절됐다.

## 왜 아직 실제 assert가 안 났는가

목표 순서는 다음과 같다.

```text
free_group open
  -> Mamba alloc miss
  -> evict Full 1,590+ tokens
  -> Full free가 deferred queue에 들어감
  -> immediate Mamba retry
  -> pre-patch assert
```

현재까지는 scheduler가 batch 처리 전에 Full cache를 먼저 줄이거나, 요청을
여러 batch로 나누거나, evictable Mamba state를 먼저 반환해 immediate retry가
성공했다.

## 다음 재현 방향

1. 여러 HTTP 요청이 확실히 하나의 prefill batch가 되도록 동시 도착 장벽을
   사용한다.
2. 긴 단일 요청을 허용하도록 `max-total-tokens`를 올린다.
3. 전체 byte pool은 커지지 않도록 다른 budget을 함께 조절한다.
4. Full donor cache는 유지하되 evictable Mamba state는 최소화한다.
5. 로그에서 `group_open=True`, Full eviction, queued Full tokens, failed retry가
   한 묶음으로 나타나는지 확인한다.
6. pre-patch에서 재현되면 같은 입력을 `f600c3dff`에 그대로 적용해 정상
   완료와 deferred-free flush를 비교한다.

## 증거 로그

```text
/tmp/sglang-prepatch-tracked-server.log
/tmp/sglang-prepatch-3branch-server.log
/tmp/sglang-prepatch-24k-server.log
```

현재 판정: **CPU 재현 성공 / 실제 SGLang serving assert 재현은 미완료**.
