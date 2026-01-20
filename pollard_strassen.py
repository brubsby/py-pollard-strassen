import sys
import math
import argparse
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

def pollard_strassen(N, B=None):
    """
    Factors N using Pollard-Strassen algorithm.
    Finds a factor p <= B (if B is given) or p <= N^(1/2).
    """
    # Convert N to fmpz string for safety if it's huge
    N_val = fmpz(str(N))
    
    print(f"Factoring N = {N_val}")

    if B:
        print(f"User specified bound B = {B}")
        # L = ceil(sqrt(B))
        # Search range is roughly [1, L^2] which covers B if L^2 >= B
        sqrt_B = math.isqrt(int(B))
        if sqrt_B * sqrt_B < int(B):
            sqrt_B += 1
        L = int(sqrt_B)
        print(f"Set L (step size) = {L} based on bound {B}")
    else:
        # 1. Determine L = ceil(N^(1/4))
        # For huge numbers, math.isqrt might take int.
        try:
            sqrt_N = math.isqrt(int(N_val))
        except OverflowError:
            # Fallback for extremely large numbers if standard int fails (unlikely in Py3)
            sqrt_N = int(N_val) // (10**(len(str(N_val))//2)) # Rough approx or use flint sqrt?
            # Flint fmpz has integer sqrt
            sqrt_N = int(N_val.isqrt())
            
        root_4_N = math.isqrt(sqrt_N)
        L = int(root_4_N + 1)
        
        print(f"Calculated L (step size) = {L} based on N^(1/4)")
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factor large integers using Pollard-Strassen.")
    parser.add_argument("N", type=str, help="The integer to factor")
    parser.add_argument("--bound", "-B", type=str, help="Search for factors up to this bound (e.g., 1000000)")
    
    args = parser.parse_args()
    
    try:
        # Check if N is decimal or hex/other
        N_str = args.N
        target = int(N_str)
        
        bound = None
        if args.bound:
            bound = int(args.bound)
            
        result = pollard_strassen(target, B=bound)
        
        if result:
            print(f"Found factor: {result}")
            print(f"Complement: {target // result}")
        else:
            print("No factor found within the search range.")
            
    except ValueError as e:
        print(f"Error: {e}")