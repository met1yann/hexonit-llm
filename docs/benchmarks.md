# Benchmarks

Community benchmark results for hexonit-llm.

> To add your benchmark, run `pipe.benchmark(runs=10)` and open a PR with your results.

## How to Benchmark

```python
from hexonit_llm import UltraInference

pipe = UltraInference("meta-llama/Meta-Llama-3-8B-Instruct")
stats = pipe.benchmark(prompt="Explain quantum computing in simple terms.", runs=10)
print(stats)
```

## Results Table

| Hardware | Engine | Model | Quant | Tok/s | Contributor |
|----------|--------|-------|-------|-------|-------------|
| _Coming soon_ | | | | | |

## Notes

- Token count is approximated as `words * 1.3`
- Results vary based on GPU, CPU, RAM, and system load
- For reproducible benchmarks, close other GPU-intensive applications