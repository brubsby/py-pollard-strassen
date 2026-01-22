import sys
import math
import argparse
import resource
import psutil
from pollard_strassen import pollard_strassen, get_memory_cost_params

def parse_memory_limit(mem_str):
    """
    Parses a memory string (e.g., '1GB', '500M') into bytes.
    """
    if not mem_str:
        return None
    
    mem_str = mem_str.upper().strip()
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    
    multiplier = 1
    for unit, value in units.items():
        if mem_str.endswith(unit) or mem_str.endswith(unit + "B"):
            multiplier = value
            # Remove unit suffix
            mem_str = mem_str.rstrip("B").rstrip(unit)
            break
            
    try:
        return int(float(mem_str) * multiplier)
    except ValueError:
        raise ValueError(f"Invalid memory format: {mem_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factor large integers using Pollard-Strassen.")
    parser.add_argument("N", type=str, help="The integer to factor")
    parser.add_argument("--bound", "-b", type=str, help="Search for factors up to this bound (e.g., 1000000)")
    parser.add_argument("--max-memory", "-m", type=str, help="Approximate max memory usage (e.g., '1GB', '512MB')")
    parser.add_argument("--free-ram-percent", "-f", type=float, default=10.0, help="Percentage of total RAM to leave free (default: 10). Set to 0 to disable.")
    parser.add_argument("--prove-smallest-factor", "-p", type=str, help="Prove that this factor is the smallest factor of N")
    
    args = parser.parse_args()
    
    try:
        # Check if N is decimal or hex/other
        N_str = args.N
        target = int(N_str)
        
        bound = None
        if args.bound:
            bound = int(args.bound)
            
        max_mem_bytes = None
        if args.max_memory:
            max_mem_bytes = parse_memory_limit(args.max_memory)

        # Calculate limit from free-ram-percent
        if args.free_ram_percent > 0:
            vm = psutil.virtual_memory()
            # (total RAM) * (1 - percent/100) - (currently used RAM)
            # This represents the remaining budget we can consume.
            target_utilization = vm.total * (1 - args.free_ram_percent / 100.0)
            calculated_limit = int(target_utilization - vm.used)
            
            if calculated_limit <= 0:
                print(f"Warning: System memory usage is already above the safety threshold (keeping {args.free_ram_percent}% free).")
                calculated_limit = 1 # Force strict limit in function
            
            if max_mem_bytes is not None:
                max_mem_bytes = min(max_mem_bytes, calculated_limit)
            else:
                max_mem_bytes = calculated_limit

        if args.prove_smallest_factor:
            prove_factor = int(args.prove_smallest_factor)
            if target % prove_factor != 0:
                print(f"Error: Factor {prove_factor} does not divide the composite!")
                sys.exit(1)
            
            # To prove prove_factor is the smallest, we must scan up to prove_factor.
            # Set bound to prove_factor.
            bound = prove_factor
            
            # Calculate required RAM
            L_req = math.isqrt(bound)
            if L_req * L_req < bound:
                L_req += 1
                
            fixed_oh, cost_per_L = get_memory_cost_params(target)
            required_mem = fixed_oh + (L_req * cost_per_L)
            
            print(f"Proof requires checking up to factor {prove_factor}.")
            print(f"Required L: {L_req}")
            print(f"Required RAM: {required_mem / 1024 / 1024:.2f} MB")
            
            if max_mem_bytes is not None and max_mem_bytes < required_mem:
                print(f"Error: Insufficient memory to prove factor {prove_factor} is the smallest.")
                print(f"Available: {max_mem_bytes / 1024 / 1024:.2f} MB")
                print(f"Required:  {required_mem / 1024 / 1024:.2f} MB")
                sys.exit(1)

        result = pollard_strassen(target, B=bound, max_memory=max_mem_bytes)
        
        if result:
            print(f"Found factor: {result}")
            print(f"Complement: {target // result}")
        else:
            print("No factor found within the search range.")
            
    except ValueError as e:
        print(f"Error: {e}")        
    # Report Peak Memory
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak Memory Usage: {usage_kb / 1024:.2f} MB")
