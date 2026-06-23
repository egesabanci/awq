# FP16 Baseline Report

- **Model:** models/Qwen3.5-2B-FP16
- **Device:** mps
- **Dtype:** float16
- **Samples:** 100

## Timing

| Metric | Value |
|--------|-------|
| Total time | 369.17 s |
| Mean throughput | 8.26 tok/s |
| Median throughput | 9.47 tok/s |
| Total tokens generated | 1354 |

## Memory

| Metric | Value |
|--------|-------|
| Peak allocated | 3.76 GB |
| Current allocated | 3.76 GB |

## Tool-Calling Quality (Self-Consistency Check)

| Metric | Mean | Std |
|--------|------|-----|
| Json Valid | 13.0% | 33.6% |
| Function Name Correct | 22.0% | 41.4% |
| Param Presence | 55.8% | 48.0% |
| Param Accuracy | 35.4% | 45.9% |
| Has Tool Call | 65.0% | 47.7% |

## Semantic Similarity (Baseline Self-Check)

| Metric | Value |
|--------|-------|
| Mean cosine | 1.0 |
| Std | 7.275674107631858e-08 |
| Min | 0.9999998807907104 |

## Sample Outputs

### Sample 1

```
<think>

</think>

[Get Reduced VAT Categories]
```

### Sample 2

```
user
<think>

</think>

[getPageSpeed(url="www.newsite.com"), Get Torrents from eztv(searchtopic="Breaking Bad")]
```

### Sample 3

```
<think>

</think>

[furnace.calculate_energy_consumption(furnace_id="F101", process_details={"material": "Steel", "treatment_type": "Annealing", "duration": {"start_time": "Morning", "hours": 5}}), furnace.calculate_energy_consumption(furnace_id="F102", process_details={"material": "Aluminum", "trea
... (truncated)
```

### Sample 4

```
<think>

</think>

[Safe For Work (SFW) Image API](Safe%20For%20Work%20(SFW)%20Image%20API)
```

### Sample 5

```
user
```
