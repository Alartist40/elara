import time
import argparse
from elara_core.tiers.tier1 import Tier1Engine
from elara_core.tiers.tier2 import Tier2Engine
from elara_core.tiers.tier3 import Tier3Engine

def benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What is the capital of France?")
    args = parser.parse_args()

    tier1 = Tier1Engine()
    tier2 = Tier2Engine(tier1)
    tier3 = Tier3Engine()

    print(f"Benchmarking query: {args.query}")

    # Tier 1
    start = time.time()
    res1 = tier1.generate(args.query)
    lat1 = (time.time() - start) * 1000
    print(f"Tier 1 Latency: {lat1:.2f}ms")

    # Tier 2 (without index)
    start = time.time()
    res2 = tier2.generate(args.query)
    lat2 = (time.time() - start) * 1000
    print(f"Tier 2 Latency: {lat2:.2f}ms")

    # Tier 3
    if tier3.is_available():
        start = time.time()
        res3 = tier3.generate(args.query)
        lat3 = (time.time() - start) * 1000
        print(f"Tier 3 Latency: {lat3:.2f}ms")
    else:
        print("Tier 3 not available (no API key)")

if __name__ == "__main__":
    benchmark()
