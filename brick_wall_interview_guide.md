# Brick Wall - Interview Walkthrough Guide

## Phase 1: Problem Clarification (1-2 minutes)

### Repeat the problem back
> "So we have a brick wall with rows of bricks. Each row has the same total width but bricks can have different widths. We need to find where to draw a vertical line to cross the minimum number of bricks. Lines through edges don't count as crossing."

### Ask clarifying questions
- "Can a row have just one brick?" → Yes
- "Are brick widths always positive integers?" → Yes
- "Is the wall guaranteed to have at least one row?" → Yes
- "Can I draw the line at the left or right edge?" → No

### Confirm constraints
```
1 ≤ n ≤ 10^4 (rows)
1 ≤ row.length ≤ 10^4 (bricks per row)
1 ≤ brick width ≤ 2^31 - 1
Total bricks ≤ 2 × 10^4
```

---

## Phase 2: Work Through Examples (2-3 minutes)

### Draw the example visually
```
Position:  0   1   2   3   4   5   6
           |   |   |   |   |   |   |
Row 0:     |[1]|[ 2 ]|[ 2 ]|[1]|      edges at: 1, 3, 5
Row 1:     |[   3   ]|[1]|[ 2 ]|      edges at: 3, 4
Row 2:     |[1]|[   3   ]|[ 2 ]|      edges at: 1, 4
Row 3:     |[  2  ]|[    4    ]|      edges at: 2
Row 4:     |[   3   ]|[1]|[ 2 ]|      edges at: 3, 4
Row 5:     |[1]|[   3   ]|[1]|[1]|    edges at: 1, 4, 5
```

### Count edges at each position
```
Position 1: 3 edges (rows 0, 2, 5)
Position 2: 1 edge  (row 3)
Position 3: 3 edges (rows 0, 1, 4)
Position 4: 4 edges (rows 1, 2, 4, 5)  ← MAXIMUM
Position 5: 2 edges (rows 0, 5)
```

### Verbalize the insight
> "If I draw at position 4, I pass through 4 edges, so I only cross 6 - 4 = 2 bricks."

---

## Phase 3: Identify the Key Insight (1-2 minutes)

### State the reframe
> "Instead of minimizing bricks crossed, I can MAXIMIZE edges hit. These are complementary."

### Write the formula
```
min_bricks_crossed = total_rows - max_edges_at_any_position
```

### Explain why this works
> "Every row either contributes an edge at position X (not crossed) or doesn't (crossed). So if we find the position with the most edges, we minimize crossings."

---

## Phase 4: Propose the Algorithm (2-3 minutes)

### High-level approach
1. Use a hash map to count edges at each x-position
2. Iterate through each row, tracking cumulative width
3. For each edge (except the last), increment count at that position
4. Find the maximum count
5. Return `rows - maxCount`

### Why hash map?
> "Positions can be sparse (width up to 2^31), so array won't work. Hash map gives O(1) access."

### Why skip the last edge?
> "The last edge is always at the wall's right boundary. We can't draw there per the problem rules."

### State complexity upfront
- **Time:** O(N) where N = total bricks
- **Space:** O(W) where W = unique edge positions (at most N-rows)

---

## Phase 5: Code It Clean (5-7 minutes)

### Write skeleton first
```cpp
int leastBricks(vector<vector<int>>& wall) {
    // 1. Count edges at each position

    // 2. Find max edges

    // 3. Return rows - maxEdges
}
```

### Fill in implementation
```cpp
int leastBricks(vector<vector<int>>& wall) {
    unordered_map<int, int> edgeCount;

    // Count edges at each position
    for (const auto& row : wall) {
        int pos = 0;
        for (int i = 0; i < row.size() - 1; i++) {  // skip last brick
            pos += row[i];
            edgeCount[pos]++;
        }
    }

    // Find max edges
    int maxEdges = 0;
    for (const auto& [pos, count] : edgeCount) {
        maxEdges = max(maxEdges, count);
    }

    return wall.size() - maxEdges;
}
```

### Talk while coding
- "I'm using an unordered_map for O(1) lookups..."
- "I stop at size()-1 to skip the right boundary..."
- "pos accumulates the x-coordinate of each edge..."

---

## Phase 6: Trace Through Example (2-3 minutes)

### Dry run with the example
```
Row [1,2,2,1]: pos=1 (count=1), pos=3 (count=1), pos=5 (count=1)
Row [3,1,2]:   pos=3 (count=2), pos=4 (count=1)
Row [1,3,2]:   pos=1 (count=2), pos=4 (count=2)
Row [2,4]:     pos=2 (count=1)
Row [3,1,2]:   pos=3 (count=3), pos=4 (count=3)
Row [1,3,1,1]: pos=1 (count=3), pos=4 (count=4), pos=5 (count=2)

maxEdges = 4
return 6 - 4 = 2 ✓
```

---

## Phase 7: Edge Cases (1-2 minutes)

### Identify and handle
| Edge Case | Example | Result |
|-----------|---------|--------|
| Single brick per row | `[[5],[5],[5]]` | 3 (no edges, must cross all) |
| All edges aligned | `[[1,1],[1,1],[1,1]]` | 0 (line at pos 1 crosses none) |
| Single row | `[[1,2,3]]` | 0 (can hit an edge) |
| Wide bricks | `[[2^30, 2^30]]` | Works (hash map handles large keys) |

### Verify single-brick case
```cpp
// If every row has 1 brick, the inner loop never runs
// edgeCount stays empty → maxEdges = 0
// return wall.size() - 0 = wall.size() ✓
```

---

## Phase 8: Optimization Discussion (1 minute)

### Is there a better approach?
> "No, we must look at every brick at least once to know where edges are. O(N) is optimal."

### Space optimization?
> "We could process rows one at a time if memory is critical, but O(W) is already good since W ≤ total bricks."

### Alternative data structures?
> "Sorted map (TreeMap) would give ordered positions but O(log W) per operation. Not needed here."

---

## Common Mistakes to Avoid

1. **Including the rightmost edge** → Line can't be at wall boundary
2. **Using array instead of hash map** → Positions can be huge (2^31)
3. **Counting bricks instead of edges** → Harder to reason about
4. **Off-by-one in loop** → Must be `i < row.size() - 1`
5. **Forgetting empty map case** → When all rows have single brick

---

## Time Budget Summary

| Phase | Time | Cumulative |
|-------|------|------------|
| Clarify | 1-2 min | 2 min |
| Examples | 2-3 min | 5 min |
| Insight | 1-2 min | 7 min |
| Algorithm | 2-3 min | 10 min |
| Code | 5-7 min | 17 min |
| Trace | 2-3 min | 20 min |
| Edge cases | 1-2 min | 22 min |
| Optimize | 1 min | 23 min |

**Buffer:** 7 minutes for questions/discussion

---

## Key Phrases for the Interview

- "Let me reframe this as a maximization problem..."
- "The key insight is that edges and crossings are complementary..."
- "I'll use a hash map because positions can be sparse..."
- "Let me trace through to verify..."
- "This handles the edge case where..."
