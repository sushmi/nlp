# NOTES
## CPU / GPU Clock

What is a Clock?
A clock is an electronic signal that synchronizes all operations in a processor. It "ticks" billions of times per second.

```
Clock Signal:
 ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐
 │  │  │  │  │  │  │  │  │  │
─┘  └──┘  └──┘  └──┘  └──┘  └──

Each tick = one "cycle"
```

Clock Speed (Frequency)
Measured in Hz (Hertz) = cycles per second

Unit	Meaning	Example
MHz	Million cycles/sec	1000 MHz = 1 billion ticks/sec
GHz	Billion cycles/sec	3.5 GHz = 3.5 billion ticks/sec

What Happens Each Tick?
Each clock cycle, the processor can:
- Fetch an instruction
- Execute a calculation
- Move data

Faster clock = More operations per second = Faster processing

### Real-World Example (Your M3 Mac)

| Component | Clock Speed     | Meaning                        |
|-----------|----------------|--------------------------------|
| M3 CPU    | ~3.0-4.0 GHz   | 3-4 billion cycles per second  |
| M3 GPU    | ~1.0-1.4 GHz   | 1-1.4 billion cycles per second|

From your powermetrics output:

```
GPU HW active frequency: 1002 MHz = 1.002 GHz
```
Your GPU was running at ~1 billion cycles per second.

### Why GPU Clock is Lower Than CPU?

| Aspect    | CPU                | GPU                        |
|-----------|--------------------|----------------------------|
| Clock     | 3-5 GHz (fast)     | 1-2 GHz (slower)           |
| Cores     | 8-24 cores         | 1000s of cores             |
| Strategy  | Few fast cores     | Many slow cores            |

GPU compensates for slower clock with massive parallelism.

Analogy:

```
CPU: 8 expert chefs cooking at high speed
     → Great for complex, sequential recipes

GPU: 1000 line cooks working together
     → Great for making 1000 identical dishes at once
```

Clock Speed vs Performance:
⚠️ Higher clock ≠ always faster

Other factors matter:
- Number of cores
- Instructions per cycle (IPC)
- Cache size
- Memory bandwidth
- That's why an M3 at 3 GHz can beat older chips at 4 GHz.

---

## Pyspark vs Pandas

| Feature               | Pandas        | PySpark         | Polars         |
|-----------------------|---------------|-----------------|---------------|
| Memory Efficiency     | Moderate      | High            | Very High     |
| Parallelism           | No            | Distributed     | Multi-threaded|
| Dataset Size Handling | Small-Medium  | Large-Very Large| Medium-Large  |
| I/O Performance       | Moderate      | High            | High          |
| Real-Time Processing  | Poor          | Excellent       | Moderate      |
| Ease of Use           | Excellent     | Moderate        | Good          |
| Columnar Optimization | Limited       | Good            | Excellent     |
