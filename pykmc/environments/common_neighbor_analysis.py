"""Determine cristalline environments."""

def cna(neighbors_list: list[list[int]]) -> list[str] : 
    """Classify atomic environments by neighbor count.

    Determine if each atom's environment is 'crystal' (12, 8, or 6 neighbors)
    or 'noncrystal'.

    Parameters
    ----------
    neighbors_list : list[list[int]]
        List of first nearest neighbor indices for each atom.

    Returns
    -------
    list[str]
        'crystal' or 'noncrystal' classification for each atom.

    """

    hash = [] 
    neighbor_sets = [set(neighbors) for neighbors in neighbors_list]
    #Compute signature
    for i, neighbors_i in enumerate(neighbors_list):
        signatures = {} #signature for all i,j pairs
        neighbors_i_set = neighbor_sets[i]
        for j in neighbors_i : 
            neighbors_j_set = neighbor_sets[j]

            #common neighbors between i and j : first signature value
            common_neighbors = neighbors_i_set & neighbors_j_set #intersection
            n_common = len(common_neighbors)
            if n_common == 0 : 
                continue 
            
            #How many common_neighbors are first neighbors/connected 
            n_bonds = 0 
            for k in common_neighbors : 
                neighbors_k_set = neighbor_sets[k]
                n_bonds += len(neighbors_k_set & common_neighbors) #Check neighbors of k in common neighbors of i and j
            n_bonds //=2 

            #Signature (n_common, n_bonds)
            sig = (n_common, n_bonds)
            signatures[sig] = signatures.get(sig, 0) +1 #counter of same signature

        #Compute hash : 
        if is_crystal(signatures) : 
            hash.append("crystal")
        else : 
            hash.append("noncrystal")
    return hash

def is_crystal(signatures:dict) : 

    if not signatures : 
        return False
    
    #FCC : when 12 signature (4,2)
    #HCP: 6 signatures (4,2,1) and 6 signatures (4,2,2) so 12 signature (4,2)
    if signatures.get((4,2), 0) == 12 : 
        return True 
    
    #BCC : 6 signature (4,4) and 8 signature (6,6) : 
    if signatures.get((4,4), 0) == 6 and signatures.get((6,6), 0) == 8 : 
        return True 
    
    #ICO : 12 signature (5,5) 
    if signatures.get((5,5), 0) == 12 : 
        return True 
    
    return False

