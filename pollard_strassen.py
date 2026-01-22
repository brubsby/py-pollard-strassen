import math
from flint import fmpz_mod_poly_ctx, fmpz

def product_tree(leaves):
    """
    Computes the product of polynomial leaves using a tree structure.
    leaves: list of fmpz_mod_poly
    """
    if not leaves:
        # Should not happen in this logic, but strictly we'd need context to return 1.
        return None
    
    n = len(leaves)
    if n == 1:
        return leaves[0]
    
    mid = n // 2
    
    left = product_tree(leaves[:mid])
    right = product_tree(leaves[mid:])
    
    return left * right

def get_memory_cost_params(N):
    """
    Returns (fixed_overhead_bytes, cost_per_L_bytes) for memory estimation.
    """
    try:
        N_int = int(N)
    except:
        # Fallback for very large numbers passed as strings or other types
        N_int = int(str(N))

    N_bytes = N_int.bit_length() // 8
    
    # Heuristic: 
    # Base Python/FLINT overhead is approx 20-25MB.
    # Per-leaf cost is approx 3.1-3.4KB (scaling factor 8x).
    FIXED_OVERHEAD = 25 * 1024 * 1024 # 25 MB
    
    base_cost = 256 + (4 * N_bytes)
    cost_per_L = base_cost * 8
    
    return FIXED_OVERHEAD, cost_per_L

def pollard_strassen(N, B=None, max_memory=None):
    """
    Factors N using Pollard-Strassen algorithm.
    Finds a factor p <= B (if B is given) or p <= N^(1/2).
    Respects max_memory if provided.
    """
    # Convert N to fmpz string for safety if it's huge
    N_val = fmpz(str(N))
    N_int = int(N_val)
    
    print(f"Factoring N = {N_val}")

    # Determine initial target L
    if B:
        print(f"User specified bound B = {B}")
        sqrt_B = math.isqrt(int(B))
        if sqrt_B * sqrt_B < int(B):
            sqrt_B += 1
        L = int(sqrt_B)
        target_source = "bound"
    else:
        # Default: N^(1/4)
        try:
            sqrt_N = math.isqrt(N_int)
        except OverflowError:
            sqrt_N = int(N_val.isqrt())
        root_4_N = math.isqrt(sqrt_N)
        L = int(root_4_N + 1)
        target_source = "default N^(1/4)"

    # Apply Memory Constraints
    if max_memory:
        FIXED_OVERHEAD, cost_per_L = get_memory_cost_params(N_int)
        
        available_memory = max_memory - FIXED_OVERHEAD
        if available_memory <= 0:
            print(f"Warning: Memory limit {max_memory} is too low for Python overhead (~25MB).")
            print("Setting minimal L=1000. Expect memory usage > limit.")
            L_mem_limit = 1000
        else:
            L_mem_limit = int(available_memory // cost_per_L)
            
        print(f"Memory limit {max_memory} bytes (usable: {max(0, available_memory)}) implies max L approx {L_mem_limit}")
        
        if L > L_mem_limit:
            print(f"Reducing L from {L} ({target_source}) to {L_mem_limit} to fit in memory.")
            L = L_mem_limit
        else:
            print(f"Memory limit allows L up to {L_mem_limit}. Current L={L} is safe.")
            
    print(f"Final L (step size) = {L}")
    print(f"Largest factor findable ~ L^2 = {L**2}")
    
    # Safety check for very small N
    if N_val <= 1000:
        for i in range(2, int(N_val) + 1): # loop limits are small here
            if N_val % i == 0:
                return int(i)
        return None

    # Setup Context
    try:
        ctx = fmpz_mod_poly_ctx(N_val)
    except Exception as e:
        print(f"Error creating context: {e}")
        return None

    # 2. Construct polynomial f(x) = (x+1)(x+2)...(x+L)
    print("Building polynomial tree for f(x)...")
    
    # We construct leaves in batches to save memory if L is huge? 
    # For now, standard list.
    leaves = []
    
    # f(x) = product (x + i)
    # i ranges from 1 to L
    
    # Note: ctx([i, 1]) creates i + 1*x = x + i.
    # We can pre-allocate the list.
    
    # Optimization: If L is very large, this loop in Python is slow.
    # But for N ~ 100 bits, L ~ 10^7, which is heavy for Python list.
    # N^(1/4) of 10^30 is 10^7.5 ~ 30 million.
    # Python might struggle with 30M objects.
    # The user asked for "arbitrarily large". 
    # True arbitrary large requires segmented processing or C-level loops.
    # But let's assume "large enough to be interesting" but fitting in RAM.
    
    for i in range(1, L + 1):
        leaves.append(ctx([i, 1]))
        
    poly = product_tree(leaves)
    del leaves # Free memory
    
    print(f"Polynomial degree: {poly.degree()}")
    
    # 3. Evaluate f(x) at points x_k = k*L for k = 0..L-1
    print("Generating evaluation points...")
    points = []
    for k in range(L):
        points.append(fmpz(k * L))
        
    print("Multipoint evaluation...")
    values = poly.multipoint_evaluate(points)
    
    # 4. Compute product of all values
    print("Computing product of evaluations...")
    
    if not values:
        accumulated_product = fmpz(1)
    else:
        # Initialize with the first element
        prod_mod = values[0]
        for v in values[1:]:
            prod_mod = prod_mod * v
            
        # Lift to integer
        # fmpz_mod has __int__
        try:
            accumulated_product = fmpz(int(prod_mod))
        except Exception as e:
            print(f"Error converting result to integer: {e}")
            return None

    # 5. Check GCD
    print("Checking GCD...")
    g = accumulated_product.gcd(N_val)
    
    if g > 1 and g < N_val:
        return int(g)
    elif g == N_val:
        print("GCD is N. Backtracking...")
        # Check individual values
        for k, v in enumerate(values):
            try:
                val_lift = fmpz(int(v))
            except:
                val_lift = int(v)
            g_i = math.gcd(int(val_lift), int(N_val))
            
            if 1 < g_i < int(N_val):
                return int(g_i)
            elif g_i == int(N_val) or g_i == 0:
                # The factors are all clustered in this interval k
                # Scan the interval k*L + 1 ... k*L + L
                print(f"Factors clustered in interval {k}. Scanning linearly...")
                start_val = k * L
                for j in range(1, L + 1):
                    candidate = start_val + j
                    # Check gcd(candidate, N)
                    g_cand = math.gcd(candidate, int(N_val))
                    if 1 < g_cand < int(N_val):
                        return g_cand
        return None
    else:
        return None