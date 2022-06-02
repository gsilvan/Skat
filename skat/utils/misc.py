def disjoint(items):
    """Return True if the flatten set of list items is disjoint."""
    union = set()
    for item in items:
        for x in item:
            if x in union:
                return False
            union.add(x)
    return True
