import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/predict"

# A representative 13-feature payload (a moderate-PM2.5 hour from the training set)
# matching the deployed model's feature schema.
payload = {
    "features": {
        "temp definition °c": 20.3,
        "dew point definition °c": 16.5,
        "rel hum definition %": 79.0,
        "wind_u": 12.267,
        "wind_v": -3.287,
        "pm25": 9.0,
        "upwind_pm25": 17.011,
        "transport_potential": -8.952,
        "wind_alignment": -0.848,
        "no": 0.012,
        "no2": 0.012,
        "nox": 0.025,
        "o3": 0.036,
    }
}

TOTAL_REQUESTS = 100


def send_request():
    start = time.perf_counter()

    try:
        response = requests.post(URL, json=payload, timeout=30)

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "success": response.status_code == 200,
            "latency": latency_ms
        }

    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "success": False,
            "latency": latency_ms
        }


def percentile(values, p):
    values = sorted(values)

    k = (len(values) - 1) * p / 100

    f = int(k)

    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


loads = [1, 5, 10, 20, 50]

print("\n===== LATENCY / LOAD BENCHMARK =====\n")

results_table = []

for workers in loads:

    start_total = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda _: send_request(),
                range(TOTAL_REQUESTS)
            )
        )

    elapsed = time.perf_counter() - start_total

    latencies = [r["latency"] for r in results]

    success_count = sum(r["success"] for r in results)

    avg_latency = statistics.mean(latencies)

    median_latency = statistics.median(latencies)

    p95_latency = percentile(latencies, 95)

    max_latency = max(latencies)

    throughput = TOTAL_REQUESTS / elapsed

    results_table.append([
        workers,
        round(avg_latency, 2),
        round(median_latency, 2),
        round(p95_latency, 2),
        round(max_latency, 2),
        round(throughput, 2),
        success_count
    ])

print(
    f"{'Users':<10}"
    f"{'Avg(ms)':<12}"
    f"{'Median':<12}"
    f"{'P95':<12}"
    f"{'Max':<12}"
    f"{'Req/s':<12}"
    f"{'Success'}"
)

print("-" * 80)

for row in results_table:
    print(
        f"{row[0]:<10}"
        f"{row[1]:<12}"
        f"{row[2]:<12}"
        f"{row[3]:<12}"
        f"{row[4]:<12}"
        f"{row[5]:<12}"
        f"{row[6]}"
    )

print("\nBenchmark complete.")