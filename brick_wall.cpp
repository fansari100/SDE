#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

class Solution {
public:
    int leastBricks(vector<vector<int>>& wall) {
        // Map to count edges at each x-position
        unordered_map<int, int> edgeCount;

        // For each row, calculate cumulative positions of brick edges
        for (const auto& row : wall) {
            int position = 0;
            // Don't count the last brick's edge (right wall boundary)
            for (int i = 0; i < row.size() - 1; i++) {
                position += row[i];
                edgeCount[position]++;
            }
        }

        // Find the maximum number of edges at any position
        int maxEdges = 0;
        for (const auto& pair : edgeCount) {
            maxEdges = max(maxEdges, pair.second);
        }

        // Minimum bricks crossed = total rows - max edges aligned
        return wall.size() - maxEdges;
    }
};
