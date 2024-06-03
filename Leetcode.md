# Leetcode

- 本文摘录 Leetcode 各算法的题目
- 其中包括
  1. 遇到什么题时应该想到该算法
  2. 该算法的细节心得
- 其中每一大类是二级标题，1个或多个大类间用 --- 分隔

---

## Python

#### defaultdict

- `defaultdict(int)`，`defaultdict(list)`
- 接受`callable`对象，所以如果想要赋默认值：`defaultdict(lambda: float('inf'))`

#### dict

- `get(item)`如果不存在返回`None`，也可以返回默认值`get(item, val)`
- `keys()`，`values()`
- `setdefault(key, default=None)`也可以用defaultdict

---

## 模拟

#### 54 螺旋矩阵

- 明确好上下左右边界即可

---

[二分大类题单](https://leetcode.cn/circle/discuss/SqopEo/)

## 二分查找

- **当需要确定一个数在数组中的位置时可以考虑**
- 需要**数组有序**
- 习惯写左闭右开的形式，维护好**不变量**
- 做题顺序
  1. 将每一次二分的 mid 处理的值称为一次**guess**。对于guess要做的判断称为check
  2. 根据guess和check的含义来确定 `if(check(guess)) { left = 还是 right = }`
  3. 循环不变量$\textcolor{red}{L-1}$ 和 $\textcolor{blue}{R}$
  4. 对于二分的上下界需要根据情况选取
  5. 最终判断我们要取的是最大的红色($\textcolor{red}{L-1}$)还是最小的蓝色($\textcolor{blue}{R}$)

#### 35、74

- 最经典的二分查找
- 循环不变量
  - **红蓝染色法**：$\ge$target 蓝色 $<$ target 红色
  - $L-1$始终是红色，$R$始终是蓝色
  - 我们希望指向最小的蓝色，即 left/right 指向的地方
  

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if target <= nums[mid]:
                right = mid
            else:
                left = mid + 1
        return left
```

#### 2563 统计公平数对的个数(1721)

- 公平数对: `i < j && lower <= nums[i] + nums[j] <= upper`
- 关键是看出来使用**二分查找**
- 不要忘了迭代器也可以加减

```cpp
class Solution {
public:
    long long countFairPairs(vector<int>& nums, int lower, int upper) {
        std::sort(nums.begin(), nums.end());
        int n = nums.size();
        long long res = 0;
        for (int j = 0; j <n; j++) {
            auto l = std::lower_bound(nums.begin(), nums.begin() + j, lower - nums[j]);
            auto r = std::upper_bound(nums.begin(), nums.begin() + j, upper - nums[j]);
            res += r - l;
        }
        return res;
    }
};
```



## 二分答案

- 除了二分查找外，二分答案也非常常见
- 找最大的最小值，最小的最大值
- 难点并不在二分，而在于 `check()`。比如周赛遇到的一题，对容斥原理有要求。
- 一般要对条件换一个视角解读，归为老题可能更有帮助

#### 275 H指数 II

- `citations[i]` 表示研究者的第 `i` 篇论文被引用的次数。返回**至少** 有 `h` 篇论文分别被引用了**至少** `h` 次。
  - 如果 `citations[n-mid] >= mid`，说明有 mid 篇文章次数$\ge$​mid，这意味着可以提高 left，因为所有小于mid的一定也符合这个性质
  - 否则说明没有，这意味着大于mid也不可能，减小right
  - 红蓝染色法：红色满足H指数的性质，蓝色不满足
    - 循环不变量：$L-1$一定是红色，$R$一定是蓝色
  - 我们希望找到最大的红色，即 $L-1$

```cpp
class Solution {
public:
    int hIndex(vector<int>& citations) {
        int left = 1, right = citations.size() + 1, n = citations.size();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (citations[n - mid] >= mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left - 1; 
    }
};
```

#### 410 分割数组的最大值

> 给定一个非负整数数组 `nums` 和一个整数 `k` ，你需要将这个数组分成 `k` 个非空的连续子数组。
>
> 设计一个算法使得这 `k` 个子数组各自和的最大值最小。求这个最大值

- 最大的最小，最小的最大应该想到二分
- 我们二分答案，答案的最小值是 max(nums)，再小就装不下了。最大值是 sum(nums) 即k=1
- `check`: 我们贪心的从左往右加入元素，直到装满 guess。最后得到 m 个连续子数组，只要 $m \le k$，那么就成立，这是因为我们可以通过进一步的拆分使$m = k$​
  - 红蓝染色法：红色不满足这个 check，蓝色满足（guess越大越容易满足check）
  - 循环不变量：$L-1$是红色，$R$是蓝色
  - 我们希望求得最小的蓝色，即 left

```cpp
class Solution {
public:
    int splitArray(vector<int>& nums, int k) {
        int n = nums.size();
        auto check = [&](int guess) {
            int m = 1, sum = 0;
            for (int i = 0; i < n; i++) {
                if (sum + nums[i] > guess) {
                    sum = nums[i];
                    m++;
                } else {
                    sum += nums[i];
                }
            }
            // m <= k 是对的，因为我们自然可以通过拆分使 m = k
            return m <= k;
        };

        int left = *max_element(nums.begin(), nums.end());
        int right = accumulate(nums.begin(), nums.end(), 0) + 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (check(mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
};
```



---

## 链表

#### 21 合并两个有序链表
- 新建一个 Dummy node 事半功倍
```cpp
ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
    ListNode *c1 = list1, *c2 = list2;
    // Dummy node
    ListNode res = ListNode();
    ListNode *cur = &res;
    while (c1 && c2) {
        if (c1->val < c2->val) {
            cur = cur->next = c1;
            c1 = c1->next;
        } else {
            cur = cur->next = c2;
            c2 = c2->next;
        }
    }
    if (c1)  cur->next = c1;
    if (c2)  cur->next = c2;
    return res.next;
}
```

#### 141 142 环形链表 I II
- 老生常谈
- 在**CS143**检测继承关系是否存在回路时用到过

#### 160 相交链表
- 时间复杂度最低就是$O(m+n)$
- 为了达到$O(1)$的内存空间，一个朴素的想法是将第一个链表所有的元素改为负的，随后遍历第二个链表，若是遇到负的元素则说明这个节点就是相交节点，随后还原第一个链表
- 一个更好的想法是利用相交链表的性质。同时遍历两个链表，短的链表指针遍历完后走长的，唱的遍历完后走短的，消除了步数差。
- 时间复杂度
    - 第一个方法遍历两遍短的链表和一遍相交节点距离长链表开头的距离
    - 第二个方法相当于遍历了两倍的短链表和相交节点距离长链表开头的距离
    - 第一个方法反而更快

#### 206 反转链表
- 尽管这些题简单，但对思维的考量并不简单
- 迭代，通过 left right 更形象轻松的写出了解法
```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *cur = head, *left = nullptr, *right;
        while (cur) {
            right = cur->next;
            cur->next = left;
            left = cur;
            cur = right;
        }
        return left;
    }
};
```
- 递归
```cpp
class Solution {
public:
    ListNode *reverseNode(ListNode *cur, ListNode *left) {
        ListNode *head;
        if (cur) {
            head = reverseNode(cur->next, cur);
            cur->next = left;
        } else {
            head = left;
        }
        return head;
    }

    ListNode* reverseList(ListNode* head) {
        return reverseNode(head, nullptr);
    }
};
```

#### 234 回文链表
- 找到中点后拆分并反转其中一个链表后进行比较
- 对于链表的中点，我们使用**快慢指针**进行寻找
- 距离上一次刷这些题已经是一年前了，相比之前，现在写题更加有**模块化思想**，写的更加清晰，思考的速度也更快了

---

[分享丨【题单】滑动窗口（定长/不定长/多指针） - 力扣（LeetCode）](https://leetcode.cn/circle/discuss/0viNMK/)

## 定长滑动窗口

#### 2461 长度为k的子数组的最大和(1553)

- 一眼滑动窗口，维护好size即可

```cpp
class Solution {
public:
    long long maximumSubarraySum(vector<int>& nums, int k) {
        int left = 0, right = 0, n = nums.size();
        long long window_sum = 0, res = 0;
        unordered_set<int> window{};
        for (; right < n; right++) {
            if (window.size() < k) {
                while (window.count(nums[right])) {
                    window_sum -= nums[left];
                    window.erase(nums[left++]);
                }
                window_sum += nums[right];
                window.emplace(nums[right]);
            }
            if (window.size() == k) {
                res = max(res, window_sum);
                window_sum -= nums[left];
                window.erase(nums[left++]);
            }
        }
        return res;
    }
};
```



#### 滑动子数组的美丽值（1786）
- 对于第x小的元素，关注到数据范围$[-50, 50]$
- 我们可以用一个map统计滑动窗口内所有数出现的次数，随后从小到大寻找第x个数
    - 时间复杂度：$O(nlogn)$

```cpp
class Solution {
public:
    vector<int> getSubarrayBeauty(vector<int>& nums, int k, int x) {
        int left = 0, right = 0, n = nums.size();
        map<int, int> times{};
        vector<int> res(n - k + 1);
        for (; right < n; right++) {
            if (right - left + 1 <= k) {
                if (times.count(nums[right]))
                    times[nums[right]]++;
                else
                    times.emplace(nums[right], 1);
            }
            if (right - left + 1 == k) {
                int t = 0;
                for (auto [i, j]: times) {
                    if (t + j < x)
                        t += j;
                    else {
                        res[left] = i < 0 ? i : 0;
                        break;
                    }
                }
                if (--times[nums[left]] == 0)
                    times.erase(nums[left]);
                left++;
            }
        }
        return res;
    }
};
```

- 这里或许先填充k个元素，随后保持窗口大小是更好理解的方式，也避免了判断的开销
- 一个更好的想法是直接用unordered_map记录，对于寻找第x个数，直接从$[-50, 0]$枚举寻找



## 不定长滑动窗口（求最长/最大）

#### 2401 最长优雅子数组

>  如果 `nums` 的子数组中位于 **不同** 位置的每对元素按位 **与（AND）**运算的结果等于 `0` ，则称该子数组为 **优雅** 子数组

- 一眼滑动窗口
- 关键是位运算

```cpp
class Solution {
public:
    int longestNiceSubarray(vector<int>& nums) {
        // 所有数全部 求或
        int orSum = nums[0], left = 0, right = 1, n = nums.size(), res = 1;
        for (; right < n; right++) {
            while (orSum & nums[right]) {
                orSum ^= nums[left];
                left++;
            }
            orSum |= nums[right];
            res = max(res, right - left + 1);
        }
        return res;
    }
};
```



## 不定长滑动窗口（求最短/最小）
#### 209 长度最小的子数组
```cpp
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int res = 1e5 + 1, left = 0, right = 0, n = nums.size();
        long long window_sum = 0;
        for (; right < n; right++) {
            window_sum += nums[right];
            while (window_sum >= target) {
                window_sum -= nums[left];
                res = min(res, right - left + 1);
                left++;
            }
        }
        return res == 1e5 + 1 ? 0 : res;
    }
};
```
- 这题求$\ge$target的最短数组，不难想到二分
```cpp
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = nums.size();
        
        auto check = [&](int guess) -> bool {
            long long window_sum = 0;
            int i = 0;
            for (; i < guess; i++)
                window_sum += nums[i];
            for (i = guess; i < n; i++) {
                if (window_sum >= target) {
                    return true;
                }
                window_sum -= nums[i - guess];
                window_sum += nums[i];
            }
            return window_sum >= target;
        };
        
        int left = 0, right = n + 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (check(mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left > n ? 0 : left;
    }
};
```



## 不定长滑动窗口（求子数组个数）

- 对于统计个数类的滑动窗口，我们需要对条件敏感，往往答案需要结合条件取滑动窗口之外的值



#### 1358 包含所有三种字符的子字符个数（1646）

- 关键是要稍微变通一下，当窗口满足要求时，窗口的右边加任何字符都满足要求，所以`res += n - right;`随后再收缩窗口

#### 2962 统计最大元素出现至少k次的子数组（1701）

- 同上题一样的思路

#### 2302 统计得分小于k的子数组数目（1808）

- 一个数组的分数定义为数组之和乘长度，返回分数严格小于k的非空子数组数目
- 当我们确定right时，我们调整窗口使窗口的子数组满足要求
- 对于这样的一对[left, right]，我们新添一个元素会获得`right - left + 1` 个新子数组。
  - 考虑极端情况，窗口为空时，`left = right + 1`，所以仍然正确

```cpp
class Solution {
public:
    long long countSubarrays(vector<int>& nums, long long k) {
        long long window_sum, res;
        int left, right, n = nums.size();
        left = right = window_sum = res = 0;
        for (; right < n; right++) {
            window_sum += nums[right];
            while (window_sum * (right - left + 1) >= k) {
                window_sum -= nums[left++];
            }
            res += right - left + 1;
        }
        return 	res;
    }
};
```

#### 统计好子数组的数目（1892）
- 同上面的思路：`res += n - right;`

```cpp
class Solution {
public:
    long long countGood(vector<int>& nums, int k) {
        unordered_map<int, int> times{};
        long long res, window_k;
        int left, right, n = nums.size();
        res = left = right = window_k = 0;

        for (; right < n; right++) {
            window_k += times[nums[right]];
            times[nums[right]]++;

            while (window_k >= k) {
                res += n - right;
                window_k -= --times[nums[left++]];
            }
        }
        return res;
    }
};
```

#### 2762 不间断子数组（1940）
- 子数组间每两个数差不超过2，同之前题的思路
```cpp
class Solution {
public:
    long long continuousSubarrays(vector<int>& nums) {
        long long res;
        int left, right, n = nums.size();
        multiset<int> num_set{};
        res = left = right = 0;

        for (; right < n; right++) {
            num_set.insert(nums[right]);
            while (*num_set.rbegin() - *num_set.begin() > 2) {
                num_set.erase(num_set.find(nums[left++]));
            }
            res += right - left + 1;
        }
        return res;
    }
};
```



---


## 动态规划
- 结合代码随想录，0-1背包，完全背包...
    - 对于 0-1 背包，关键是对于滚动数组，我们从后向前遍历才能正确的使用上一次的状态
- 对于完全背包，要分清两个循环哪个在外
    - 如果背包容量在外，说明在一个确定的容量，可以选择任意的物品，所以求出的是排列
    - 如果物品在外，说明物品的顺序不会改变，所以是组合
- 对于一般的dp题，关键还是想到合适的dp含义。如果找到合适的dp，一般递推关系都不难列出

## 入门dp
### 爬楼梯

- 记忆化搜索->递推
  - 比如打家劫舍：$dfs(i) = max(dfs(i-1), dfs(i-2)+nums[i]);$
- 其实就是完全背包。。
#### 377 组合总和 IV
- 找出和为target的组合个数，完全背包求组合数，背包容量在外即可

#### 2466 统计构造好字符串的方案数（1694）
- 0可以加zero次，1可以加one次，求长度在low和high间的字符串个数

```python
class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        dp = [0 for _ in range(high + 1)]
        dp[0], res = 1, 0
        for j in range(1, high + 1):
            for i in (zero, one):
                if j - i >= 0:
                    dp[j] += dp[j - i]
        return sum(dp[low:]) % (10 ** 9 + 7)
```

### 最大子数组和
- 两种做法：
    1. $f(i)$为以$a[i]$结尾的最大子数组和。状态转移方程：$f(i)=max(f[i-1],0)+a[i]$
    2. 前缀和

#### 1191 k次串联后最大子数组和（1748）

- 简单的算 n * k 次会超时
- 我们需要思考最大子数组和可能出现的位置
  - 出现在单个数组中，比如 [-100, 1, -100]
  - 出现在两个数组交接处，比如 [10, -100, 10]
  - 若数组之和大于0，则可以看作两个数组的交接处加上(k-2)个数组之和

```cpp
class Solution {
public:
    const int mod = (int)pow(10, 9) + 7;
    int kConcatenationMaxSum(vector<int>& arr, int k) {
        long long n = arr.size(), res = 0, dp_j = 0, dp_prevj = 0;
        for (int j = 0; j < n; j++) {
            dp_j = (dp_prevj < 0 ? 0 : dp_prevj) + arr[j];
            dp_prevj = dp_j;
            res = max(res, dp_j);
        }

        if (k == 1) return res;
        long long sum = 0;
        for (int j = 0; j < n; j++) {
            dp_j = (dp_prevj < 0 ? 0 : dp_prevj) + arr[j];
            dp_prevj = dp_j;
            res = max(res, dp_j);
            sum += arr[j];
        }

        if (sum <= 0)
            return res;
        return (res + (k - 2) * sum) % mod;
    }
};
```

#### 918 环形子数组的最大和（1777）

- 最大和要么平凡的出现在单个数组，要么出现在数组边界。
- 我们求[0:j]的最大前缀和 $f(j)=max(f(j-1),sum(arr[0:j+1]))$

```cpp
class Solution {
public:
    int maxSubarraySumCircular(vector<int>& nums) {
        int n = nums.size(), res, dp_j, sum;
        vector<int> prefixSum(n);
        res = dp_j = sum = prefixSum[0] = nums[0];
        for (int i = 1; i < n; i++) {
            dp_j = max(dp_j, 0) + nums[i];
            res = max(res, dp_j);

            sum += nums[i];
            prefixSum[i] = max(prefixSum[i - 1], sum);
        }
        sum = 0;
        for (int i = n - 1; i >= 1; i--) {
            sum += nums[i];
            res = max(res, sum + prefixSum[i - 1]);
        }
        return res;
    }
};
```





## 网格图DP

#### 2684 矩阵中移动的最大次数（1626）

- 移动的最大次数，`dfs/bfs`解决
- 一种取巧的方法就是将元素赋值为0或者相反数来标记当前元素已经走过了
- `function` 允许我们递归的使用，而lambda不行

```cpp
class Solution {
public:
    int maxMoves(vector<vector<int>>& grid) {
        int res = 0, m = grid.size(), n = grid[0].size();
        function<void(int, int)> dfs = [&](int i, int j) {
            res = max(res, j);
            if (res == n - 1)
                return;
            for (int k = max(0, i - 1); k < min(m, i + 2); k++) {
                if (grid[k][j + 1] > grid[i][j])
                    dfs(k, j + 1);
            }
            grid[i][j] = 0;
        };
        
        for (int i = 0; i < m; i++)
            dfs(i, 0);
        return res;
    }
};
```

#### 1594 矩阵的最大非负积（1807）

- 简单的dp
- 对于大的数，我们可以用 long long 来处理
- 区分 `grid[i][j]` 与 0 的关系即可

#### 1301 最大得分的路径数目（1853）

```cpp
class Solution {
public:
    int mod = (int)pow(10, 9) + 7;
    vector<int> pathsWithMaxScore(vector<string>& board) {
        int m = board.size(), n = board[0].size();
        vector<vector<int>> points(m, vector<int>(n));
        vector<vector<int>> times(m, vector<int>(n));
        board[0][0] = board[m - 1][n - 1] = '0';
        times[0][0] = 1;
        for (int j = 1; j < n; j++) {
            if (board[0][j] == 'X') break;
            points[0][j] = points[0][j - 1] + board[0][j] - '0';
            times[0][j] = 1;
        }
        for (int i = 1; i < m; i++) {
            if (board[i][0] == 'X') break;
            points[i][0] = points[i - 1][0] + board[i][0] - '0';
            times[i][0] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (board[i][j] == 'X')
                    continue;
                int maxPoint = -1, time = 0;
                for (auto [ii, jj]: vector<pair<int, int>>{{-1, 0}, {0, -1}, {-1, -1}}) {
                    if (times[i + ii][j + jj] != 0) {
                        int temp = points[i + ii][j + jj] + board[i][j] - '0';
                        if (maxPoint == temp) {
                            time += times[i + ii][j + jj];
                        } else if (maxPoint < temp) {
                            time = times[i + ii][j + jj];
                            maxPoint = temp;
                        }
                    }
                }
                if (maxPoint != -1) {
                    points[i][j] = maxPoint;
                    times[i][j] = time % mod;
                }
            }
        }
        if (times[m - 1][n - 1] == 0)
            return vector<int>{0, 0};
        return vector<int>{points[m - 1][n - 1], times[m - 1][n - 1]};
    }
};
```

#### 矩阵中的最长递增路径

- dfs

#### 85 最大矩形

- 和 84 想通， 对于第 j 列，将其看作一个横向的柱形图，求最大矩形
- `dp[i][j]:`下标(i, j)左边的最长连续1个数
- 自然的想到递推公式
- 单调栈中摘录了更快的方法

```cpp
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        int i, j, res = 0;
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    dp[i][j] = (j > 0 ? dp[i][j-1] + 1: 1);
                }
                int width = 201;
                for (int k = i; k >= 0; k--) {
                    width = min(width, dp[k][j]);
                    if (width == 0)
                        break;
                    res = max(res, width * (i - k + 1));
                }
            }
        }
        return res;
    }
};
```



## 背包

### 完全背包
#### 322 零钱兑换
- 求的是最少多少硬币，所以谁在外都可
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        res = [1e5 for i in range(amount + 1)]
        res[0] = 0
        n = len(coins)
        for i in range(1, amount + 1):
            for j in range(n):
                if i - coins[j] >= 0:
                    res[i] = min(res[i], res[i - coins[j]] + 1)
        return res[amount] if res[amount] != 1e5 else -1
```



## 经典线性dp

### 最长公共子序列

#### 1143 最长公共子序列

- 非常经典，$dp[i][j]$表示$text1[:i]$和$text2[:j]$的最长公共子序列
  - dp 是递增的
  - $text1[i] == text2[j]$ 时无需考虑 $dp[i-1][j]$ 和 $dp[i][j-1]$
    - 这是因为 $dp[i-1][j] <= dp[i-1][j-1]+1$
    - 这可以通过分类讨论来证明。讨论i-1处和j-1处公共子序列的结尾地点
- $dp[i][j] = (text1[i] == text2[j] ? dp[i-1][j-1]+1 : max(dp[i-1][j], dp[i][j-1]))$

#### 72 编辑距离

- 非常经典，见证着它难度从困难降低到中等

$$
dp[i][j] = \left\{
\begin{aligned}
dp[i-1][j-1], \quad&text1[i]==text2[j] \\
min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]), \quad&else
\end{aligned}
\right.
$$

####  1035 不相交的线（1806）

- 思路很相似

$$
dp[i][j] = \left\{
\begin{aligned}
dp[i-1][j-1] + 1, \quad&nums1[i]==nums2[j] \\
max(dp[i-1][j],dp[i][j-1]), \quad&else
\end{aligned}
\right.
$$

#### 1458 两个子序列的最大点积

- 很抽象，然而确实是对的
- 考虑所有情况，比如只选当前的(i, j)做点积，或者结合上之前的做点积

$$
dp[i][j] = max(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]+nums1[i]*nums[j],nums1[i]*nums2[j])
$$





### 最长递增子序列
#### 300 最长递增子序列
- $O(n^2)$的方法很显然
- $O(nlgn)$
    - 基于贪心，拿小的元素前去替换总是没有损失的
```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        std::vector<int> sequence{nums[0]};
        int n = nums.size();
        for (int i = 1; i < n; i++) {
            auto iter = 
            // not less than nums[i]
            std::lower_bound(sequence.begin(),sequence.end(),nums[i]);
            if (iter == sequence.end()) {
                sequence.emplace_back(nums[i]);
            } else {
                *iter = nums[i];
            }
        }
        return sequence.size();
    }
};
```

#### 32 最长有效括号

- 只想出来 $O(n^2)$的多维动态规划解法～
- 应当是潜意识受到了最长回文子串的影响
- `dp[j]`: 以s[j]结尾的最长子串

```python
# 未考虑边界check
if dp[j - 1] == '(':
    dp[j] = dp[j - 1] + 2
else:
    if dp[j - 1 - dp[j - 1]] == '(': 
        dp[j] = dp[j - 1] + 2 + dp[j - 1 - dp[j - 1]]
```

- python字符串索引即使为负也不会报错，默认从尾部开始索引



## 状态机DP

- 一般定义$f[i][j]$表示前缀$a[:i]$在状态$j$下的最优值。一般$j$​都很小

#### 121 买卖股票的最佳时机

- 只能交易一次或不交易
- 定义$f[i][j]$表示最大利润，$f[i][0]$表示第$i$天手中持有股票，$f[i][1]$表示手中无股票，即已经卖出了

$$
f[i][0] =& max(f[i-1][0], -prices[i])  \\
f[i][1] =& max(f[i-1][1], f[i-1][0]+prices[i])
$$

- 当然这题直接$f[i]=price[i]-min(price[0:i])$更快

#### 122 买卖股票的最佳时机II

- 可以买卖多次，也可以同一天买卖

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>> dp(n, vector<int>(2));
        dp[0][0] = -prices[0];
        for (int i = 1; i < n; i++) {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
        }
        return dp[n - 1][1];
    }
};
```

**请注意，我们可以去掉dp的第一个维度，如果可以去掉下面也都写去掉的版本**

- 它拥有更好的局部性以及更小的空间占用

#### 123 买卖股票的最佳时机III

- 两笔买卖
- 见下题扩展

#### 188 买卖股票的最佳时机IV

- 偶数就是第某笔交易持有股票，奇数就是卖出了
- 注意我们需要**倒着更新**

```cpp
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int i, n = prices.size();
        int dp[2 * k];
        memset(dp, 0, sizeof(dp));
        for (i = 0; i < 2 * k; i += 2)
            dp[i] = -prices[0];
        for (i = 1; i < n; i++) {
            for (int j = 2 * k - 1; j >= 1; j--) {
                if (j % 2 == 0)
                    dp[j] = max(dp[j], dp[j - 1] - prices[i]);
                else
                    dp[j] = max(dp[j], dp[j - 1] + prices[i]);
            }
            dp[0] = max(dp[0], -prices[i]);
        }
        int res = 0;
        for (i = 1; i < 2 * k; i += 2)
            res = max(res, dp[i]);
        return res;
    }
};
```

#### 309 买卖股票的最佳时机（冷冻期）

- 用vector的话可以多开一点空间省去麻烦
- 数组写稍微有点抽象，最好还是对照着原始版本理解

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        // vector<vector<int>> dp(n + 1, vector<int>(2));
        // dp[1][0] = -prices[0];
        // for (int i = 2; i <= n; i++) {
            // dp[i][0] = max(dp[i - 1][0], dp[i - 2][1] - prices[i - 1]);
            // dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i - 1]);
        // }
        // return dp[n][1];
        int dp[2] = {0}, pre = 0, tmp;
        dp[0] = -prices[0];
        for (int i = 1; i < n; i++) {
            tmp = pre;
            pre = dp[1];
            dp[1] = max(dp[1], dp[0] + prices[i]);
            dp[0] = max(dp[0], tmp - prices[i]);
        }
        return dp[1];
    }
};
```

#### 714 买卖股票的最佳时机（手续费）

- 利润减去手续费即可，最后答案$\max(dp[1], 0)$

#### 1567 乘积为正数的最长子数组长度（1710）

- 在这类动态规划里，关键总是合理的赋予$dp[i]$一个含义。在子数组问题中常见的就是，以$i$结尾的子数组（并在过程中统计结果），或者到$i$这一段满足某条件的子数组
- 这题是前者，post表示以$i$结尾的子数组的长度

```cpp
class Solution {
public:
    int getMaxLen(vector<int>& nums) {
        int n = nums.size();
        int post = 0, nega = 0, res = 0;
        if (nums[0] > 0)
            res = post = 1;
        else if (nums[0] < 0)
            nega = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] > 0) {
                post++;
                nega = nega > 0 ? nega + 1 : 0;
            } else if (nums[i] < 0) {
                int temp = post;
                post = nega > 0 ? nega + 1 : 0;
                nega = temp + 1;            
            } else {
                post = 0;
                nega = 0;
            }
            res = max(res, post);
        }
        return res;
    }
};
```

#### 2826 将三个组排序 1721

- 只有123，最少删除多少次使数组递增
- 正难则反，可以考虑最长递增子序列，这样的话时间复杂度$O(nlogn)$
- $f[i+1][j]$表示$nums[0]$到$nums[i]$且$nums[i]$变成$j$的最小修改次数
  - $f[i+1][j]=\min_{0\le k\le j}f[i][k]+nums[i]!=j$
  - 可以覆盖第一个维度，这时需要倒序遍历

```cpp
class Solution {
public:
    int minimumOperations(vector<int>& nums) {
        int dp[3] = {0};
        for (auto i: nums) {
            for (int j = 2; j >= 0; j--) {
                dp[j] = *min_element(dp, dp + j + 1) + (i != (j + 1)); 
            }
       }
       return *min_element(dp, dp + 3);
    }
};
```

## 划分型DP

### 判定能否划分

#### 139 单词拆分

- 完全背包，一个单词可以出现多次，所以是排列

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        l = len(s)
        n = len(wordDict)
        res = [0 for i in range(l + 1)]
        res[0] = 1
        for i in range(l + 1):
            for j in range(n):
                l_word = len(wordDict[j])
                if i - l_word >= 0 and res[i - l_word] == 1 and \
                s[i - l_word:i] == wordDict[j]:
                    res[i] = 1
        return res[l] == 1
```

#### 2369  检查数组是否存在有效划分（1780）

- 依次判断三种划分方式以及前面的是否已经匹配上即可

### 计算划分个数

- 枚举最后一个子数组的左端点$L$并从$f[L]$转移到$f[i]$

#### 132 分割回文串II

- 给定一个字符串s，求最少分割次数使每一个字符串都是回文串
- 先使用之前的方法确定哪些子串是回文串
- 随后定义$f[i]$表示以$i$结尾的最少分割次数
  - 我们枚举$j$来，如果$dp[j][i]$为1，我们就完成了状态转移$f[i]=min(f[i],f[j] + 1)$

```cpp
class Solution {
public:
    int minCut(string s) {
        int n = s.size();
        vector<vector<bool>> dp(n, vector<bool>(n));
        for (int i = 0; i < n; i++) dp[i][i] = true;
        for (int k = 1; k < n; k++) {
            for (int i = 0; i < n; i++) {
                int right = i + k;
                if (right >= n)
                    break;
                if (s[i] == s[right]) {
                    if (k == 1) dp[i][right] = true;
                    else
                        dp[i][right] = dp[i + 1][right - 1];
                }
            }
        }
        vector<int> f(n + 1, 2001);
        f[0] = -1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (dp[j][i])
                    f[i + 1] = min(f[i + 1], f[j] + 1);
            }
        }
        return f[n];
    }
};
```

#### 2707 字符串中的额外字符（1736）
- 不选：$f[i]=f[i-1]+1$ | 选（如果匹配）：$f[i]=min(f[i],f[i-l])$
- 这里为了避免边界的讨论，$dp[i]$表示$s[0:i]$中的额外字符数

```python
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [i for i in range(n + 1)]
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1
            for w in dictionary:
                l = len(w)
                if i - l < 0:
                    continue
                if s[i - l:i] == w:
                    dp[i] = min(dp[i], dp[i - l])
        return dp[n]
```

#### 2767 将字符串分割为最少的美丽子字符串（1865）
- 几乎和上题一样的思路

```python
pow5 = [bin(5 ** i)[2:] for i in range(7)]

class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [1e5] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            for j in pow5:
                l = len(j)
                if i - l < 0:
                    break
                if s[i - l:i] == j:
                    dp[i] = min(dp[i], dp[i - l] + 1)
        return -1 if dp[n] == 1e5 else dp[n]
```

#### 1416 恢复数组（1920）
- 关键是剪枝，注意到$1\le k\le 10^9$，我们便可以只考虑$i$的前10位

```python
class Solution:
    def numberOfArrays(self, s: str, k: int) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(n):
            for j in range(i, max(-1, i - 10), -1):
                tmp = int(s[j: i + 1])
                if tmp > k:
                    break
                if s[j] != '0' and tmp <= k:
                    dp[i + 1] += dp[j]
            dp[i + 1] %= (10 ** 9 + 7)
        return int(dp[n])

```

### 约束划分个数

- 定义$f[i][j]$表示长为$j$的前缀分成$i$​个子数组有关的最优值
- 用`dfs`可能是更直观的想法，并且记忆化后时间跟`dp`差不多

#### 410 分割数组的最大值

- 二分最快，这里用dp

```python
class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[inf] * (k + 1) for i in range(n + 1)]
        sum = [0] * (n + 1)
        for i in range(n):
            sum[i + 1] = sum[i] + nums[i]
        for i in range(k + 1):
            dp[0][i] = 0
        for i in range(n):
            for t in range(1, k + 1):
                for j in range(i, -1, -1):
                    dp[i + 1][t] = min(dp[i + 1][t], 
                    max(dp[j][t - 1], sum[i + 1] - sum[j]))
        return dp[n][k]
```

#### 1043 分割数组以得到最大和（1916）

- $dp[i] = \max(dp[i], dp[j-1]+\max(arr[j:i+1]*(i-j+1)))$

```rust
impl Solution {
    pub fn max_sum_after_partitioning(arr: Vec<i32>, k: i32) -> i32 {
        let n = arr.len();
        let mut dp = vec![0; (n + 1)];
        let k = k as usize;
        // dp[i] 代表 i-1 位置为数组最右侧
        for i in 1..=n {
            let mut mmax = 0;
            for j in 0..k.min(i) {
                mmax = mmax.max(arr[i - 1 - j]);
                dp[i] = dp[i].max(dp[i - 1 - j] + mmax * (j as i32 + 1));
            }
        }
        dp[n]
    }
}
```

```python
class Solution:
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        n = len(arr)
        @cache
        def dfs(i):
            if i == n: return 0

            mmax = res = -1
            for r in range(i, min(n, i + k)):
                mmax = max(mmax, arr[r])
                res = max(res, mmax * (r - i + 1) + dfs(r + 1))
            return res
        return dfs(0)
```

#### 813 最大平均值和的分组（1937）

- $dp[i][t]=\max_{1\le j\le i}(dp[i][t], dp[j][t-1]+mean(nums[j:i+1]))$
- 这题关键的是要**初始化**，默认的方式获得的并不是前$i$个数的均值 

```python
class Solution:
    def largestSumOfAverages(self, nums: List[int], k: int) -> float:
        n = len(nums)
        dp = [[0] * k for i in range(n + 1)]
        sum = [0] * (n + 1)
        for i in range(n):
            sum[i + 1] = sum[i] + nums[i]
            dp[i + 1][0] = sum[i + 1] / (i + 1)
        for i in range(n):
            for t in range(1, k):    
                for j in range(i + 1):
                    mean = (sum[i + 1] - sum[j]) / (i - j + 1)
                    dp[i + 1][t] = max(dp[i + 1][t], dp[j][t - 1] + mean)
        return dp[n][k - 1]

# dfs
class Solution:
    def largestSumOfAverages(self, nums: List[int], k: int) -> float:
        n = len(nums)
        preSum = [0] * n
        preSum[0] = nums[0]
        for i in range(1, n):
            preSum[i] = preSum[i - 1] + nums[i]
        # preSum[-1] = 0
        preSum.append(0)

        @cache
        def dfs(i, t):
            if i == n:
                return -inf
            if t == 1:
                return (preSum[n - 1] - preSum[i - 1]) / (n - i)
            res = 0
            for j in range(i, n):
                res = max(res, (preSum[j] - preSum[i - 1]) / (j - i + 1) + dfs(j + 1, t - 1))
            return res
        return dfs(0, k)

```

#### 1473 粉刷房子III（2056）

> - `houses[i]`表示第i个房子的颜色，0表示房子未被涂色。
>
> - `cost[i][j]`表示将房子涂成颜色 j+1 的花费
>
> 连续颜色的房子为一个街区。返回涂色使房子构成 target 个街区的最小花费

- 记忆化搜索而非dp更加简单
  - 其实二者等价，这题用到三维dp

```python
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        # color: prev house color
        @cache
        def dfs(i, color, t):
            if i == m:
                return 0 if t == target else inf
            if t > target: return inf
            
            res = inf
            if houses[i] != 0:
                res = min(res, dfs(i + 1, houses[i], t + (houses[i] != color)))
            else:
                # cur house color
                for j in range(n):
                    if j + 1 == color:
                        res = min(res, cost[i][j] + dfs(i + 1, color, t))
                    else:
                        res = min(res, cost[i][j] + dfs(i + 1, j + 1, t + 1))
            return res
        ans = dfs(0, -1, 0)
        return ans if ans != inf else -1
```

#### 2478 完美分割的方案数（2478）

> - `s` 被分成`k`个字符串。并且每个子字符串开头为质数数字，末尾不是。求多少划分方案
>   - 2/3/5/7 是质数数字
> - 每个子字符串长度至少为`minLength`

- 我们用前缀和来加速，否则*tle*

```python
class Solution:
    def beautifulPartitions(self, s: str, k: int, minLength: int) -> int:
        n = len(s)
        def isPrime(c):
            if c in '2357':
                return True
            return False

        np = []
        for i in range(n):
            if not isPrime(s[i]):
                np.append(i)
        m = len(np)

        preSum = [[0] * (k + 1) for _ in range(m + 1)]
        @cache
        def dfs(i, t):
            if i >= n:
                if t == k:
                    return 1
                return 0

            if n - i < k - t: return 0
            if not isPrime(s[i]): return 0
            if i + minLength - 1 >= n: return 0
            if t == k: return 0

            r = bisect_left(np, i + minLength - 1)
            if preSum[r][t + 1] == 0:
                for j in range(r, m):
                    tmp = dfs(np[j] + 1, t + 1)
                    preSum[j][t + 1] = preSum[j - 1][t + 1] + tmp
# res += sum([dfs(np[j] + 1, t + 1) for j in range(r, m)]) % (10 ** 9 + 7)
            res = preSum[m - 1][t + 1] - preSum[r - 1][t + 1]
            return res % (10 ** 9 + 7)
        return dfs(0, 0)
```





#### 3117 划分数组得到最小值之和（2735）

> 数组的 **值** 等于该数组的 **最后一个** 元素。
>
> 你需要将 `nums` 划分为 `m` 个 **不相交的连续** 子数组，对于第 `ith` 个子数组 ，子数组元素的按位`AND`运算结果等于 `andValues[i]`
>
> 返回将 `nums` 划分为 `m` 个子数组所能得到的可能的 **最小** 子数组 **值** 之和。如果无法完成这样的划分，则返回 `-1` 。

- 393周赛第四题，`dfs`思路跟粉刷房子很像

```python
class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        @cache
        def dfs(i, and_val, t):
            if t == m:
                return 0 if i == n else inf
           	if i >= n:
                return inf

            and_val &= nums[i]
            if and_val < andValues[t]:
                return inf
            res = dfs(i + 1, and_val, t)
            if and_val == andValues[t]:
                res = min(res, nums[i] + dfs(i + 1, -1, t + 1))
            return res
        ans = dfs(0, -1, 0)
        return ans if ans != inf else -1
```

## 区间DP

### 回文系列

#### 516 最长回文子序列

- 子序列指可以删除某个字符的子字符串，自然想到编辑距离

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for l in range(2, n + 1):
            for i in range(n - l + 1):
                r = i + l - 1
                if s[i] == s[r]:
                    if l == 2:
                        dp[i][r] = 2
                    else:
                        dp[i][r] = 2 + dp[i + 1][r - 1]
                else:
                    dp[i][r] = max(dp[i + 1][r], dp[i][r - 1])
        return dp[0][n - 1]
```

#### 1312 让字符串成为回文子串的最少插入次数（1787）

- 一开始的想法是枚举中心点，找左右子串的最长公共子串。但会tle
- 法一：找整个字符串和反字符串的最长公共子串
- 法二：`dp[i][j]`表示`i`到`j`子串的最少插入次数
  - 法二快一倍

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        n = len(s)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n):
            l = i
            for j in range(n):
                r = n - 1 - j
                if s[l] == s[r]:
                    dp[i + 1][j + 1] = 1 + dp[i][j]
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return n - dp[-1][-1]
```

## 状压DP

### 排列型（相邻无关）

- 暴力复杂度为$O(n\cdot n!)$，通常可以解决$n\le 10$的问题
- 通过状压，可以达到$O(n\cdot 2^n)$，通常可以解决$n\le 20$的问题

#### 526 优美的排列

> 假设有从 1 到 n 的 n 个整数。用这些整数构造一个数组 `perm`（**下标从 1 开始**），只要满足下述条件 **之一** ，该数组就是一个 **优美的排列** ：
>
> - `perm[i]` 能够被 `i` 整除
> - `i` 能够被 `perm[i]` 整除
>
> 给你一个整数 `n` ，返回可以构造的 **优美排列** 的 **数量** 

- 1的个数表示当前在填第几位
- 对于`mask`，比如[0, 1, 1]，表示当前已经填了两位数，分别为1和2
  - `dp[mask]`表示`mask`对应多少种填法
  - 我们枚举`mask`中的1，表示当前最后一位填什么
  - `dp[mask] += dp[m - 1 << x]`

```python
class Solution:
    def countArrangement(self, n: int) -> int:
        dp = [0] * (1 << n)
        dp[0] = 1

        for m in range(1, 1 << n):
            l = m.bit_count()
            for x in range(n):
                if m & (1 << x) and ((l % (x + 1) == 0) or ((x + 1) % l == 0)):
                    dp[m] += dp[m - (1 << x)]
        return dp[-1]

#dfs
class Solution:
    def countArrangement(self, n: int) -> int:
        option = defaultdict(list)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i % j == 0 or j % i == 0:
                    option[i].append(j)
        
        res = 0
        visited = set()
        def dfs(i):
            nonlocal res
            if i == n + 1:
                res += 1
            for x in option[i]:
                if x not in visited:
                    visited.add(x)
                    dfs(i + 1)
                    visited.discard(x)
        dfs(1)
        return res
```

#### 1879 两个数组最小的异或值之和（2145）

> 给你两个整数数组 `nums1` 和 `nums2` ，它们长度都为 `n` 。
>
> 两个数组的 **异或值之和** 为 `(nums1[0] XOR nums2[0]) + (nums1[1] XOR nums2[1]) + ... + (nums1[n - 1] XOR nums2[n - 1])` （**下标从 0 开始**）。
>
> - 比方说，`[1,2,3]` 和 `[3,2,1]` 的 **异或值之和** 等于 `(1 XOR 3) + (2 XOR 2) + (3 XOR 1) = 2 + 0 + 2 = 4` 。
>
> 请你将 `nums2` 中的元素重新排列，使得 **异或值之和** **最小** 。
>
> 请你返回重新排列之后的 **异或值之和** 。

- 观察范围$1\le n\le 14$
- 状压dp，思路同上

```python
class Solution:
    def minimumXORSum(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        dp = [inf] * (1 << n)
        dp[0] = 0
        for m in range(1, 1 << n):
            l = m.bit_count()
            for i in range(n):
                if m & (1 << i):
                    dp[m] = min(dp[m], dp[m - (1 << i)] + (nums1[l - 1] ^ nums2[i]))
        return dp[-1]
```

#### 2845将石头分散到网格图的最少移动次数

> 给你一个大小为 `3 * 3` ，下标从 **0** 开始的二维整数矩阵 `grid` ，分别表示每一个格子里石头的数目。网格图中总共恰好有 `9` 个石头，一个格子里可能会有 **多个** 石头。
>
> 每一次操作中，你可以将一个石头从它当前所在格子移动到一个至少有一条公共边的相邻格子。
>
> 请你返回每个格子恰好有一个石头的 **最少移动次数** 。

- 石头总数为9，**将问题转化**，统计石头数为0的格子为`L`，石头数大于1的格子为`R`
- 我们要做的就是找到一种`L`和`R`的匹配使移动次数最小

```python
class Solution:
    def minimumMoves(self, grid: List[List[int]]) -> int:
        L, R = [], []
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 0:
                    L.append((i, j))
                else:
                    for k in range(2, grid[i][j] + 1):
                        R.append((i, j))
        def cost(x, y):
            return abs(x[0] - y[0]) + abs(x[1] - y[1])
        n = len(L)
        dp = [inf] * (1 << n)
        dp[0] = 0
        for m in range(1, 1 << n):
            l = m.bit_count()
            for j in range(n):
                if m & (1 << j):
                    dp[m] = min(dp[m], dp[m - (1 << j)] + cost(L[l - 1], R[j]))
        return dp[-1]
```

#### 1947 最大兼容性评分和

- 老师和学生的匹配问题，同之前直接状压DP

#### 1799 N次操作后的最大分数和

> 给你 `nums` ，它是一个大小为 `2 * n` 的正整数数组。你必须对这个数组执行 `n` 次操作。
>
> 在第 `i` 次操作时（操作编号从 **1** 开始），你需要：
>
> - 选择两个元素 `x` 和 `y` 。
> - 获得分数 `i * gcd(x, y)` 。
> - 将 `x` 和 `y` 从 `nums` 中删除。
>
> 请你返回 `n` 次操作后你能获得的分数和最大为多少

- 思路差不多，关键要灵活变通

```python
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (1 << n)
        for m in range(1, 1 << n):
            l = m.bit_count()
            if l % 2 != 0: continue
            for i in range(n):
                if m & (1 << i):
                    for j in range(i + 1, n):
                        if m & (1 << j):
                            dp[m] = max(dp[m], dp[m - (1 << i) - (1 << j)] + l * gcd(nums[i], nums[j]) // 2)
        return dp[-1]
```

#### 2172 数组的最大与和（2392）

> 给你一个长度为 `n` 的整数数组 `nums` 和一个整数 `numSlots` ，满足`2 * numSlots >= n` 。总共有 `numSlots` 个篮子，编号为 `1` 到 `numSlots` 。
>
> 你需要把所有 `n` 个整数分到这些篮子中，且每个篮子 **至多** 有 2 个整数。一种分配方案的 **与和** 定义为每个数与它所在篮子编号的 **按位与运算** 结果之和。
>
> 请你返回将 `nums` 中所有数放入 `numSlots` 个篮子中的最大与和

- 合理的想法就是用`2 * numSlots`个数字来表示状态
- 那么该如何列dp方程呢
  - 我们要跳出之前的思路，寻找0的位置

```python
class Solution:
    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        n = len(nums)
        m = 2 * numSlots
        dp = [0] * (1 << m)

        for mask, dpv in enumerate(dp):
            l = mask.bit_count()
            if l >= n:  continue
            for i in range(m):
                if (mask & (1 << i)) == 0:
                    t = mask | (1 << i)
                    dp[t] = max(dp[t], dpv + ((i // 2 + 1) & nums[l]))
        return max(dp)
```

### 排列型（相邻相关）

- 一般定义$f[S][i]$表示已经排列好的元素下标集合为$S$，且上一个填的元素为$i$时和题目有关的最优值



## 树形DP

### 树的直径

#### 2246 相邻字符不同的最长路径（2126）

- `dfs(node)`为从`node`出发的最长路径
- `res`的存在很巧妙

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        n = len(parent)
        child = [[] for _ in range(n)]
        for i in range(1, n):
            child[parent[i]].append(i)
        
        ans = 0
        def dfs(node: int) -> int:
            res = 0
            nonlocal ans
            for t in child[node]:
                tmp = dfs(t) + 1
                if s[node] != s[t]:
                    ans = max(ans, res + tmp)
                    res = max(res, tmp)
            return res
        dfs(0)
        return ans + 1
```

#### 1617 统计子树中城市间的最大距离（2309）

- 每个`mask`代表一个子树，如果位为1就代表该位在子树中
- 从最高位开始`dfs`，如果走完`mask`就找到了一个直径
  - 利用`first`和`second`，我们不用担心最高位在不在直径的两个端点上

```python
class Solution:
    def countSubgraphsForEachDiameter(self, n: int, edges: List[List[int]]) -> List[int]:
        child = [[] for _ in range(n)]
        for [u, v] in edges:
            u, v = u - 1, v - 1 
            child[u].append(v)
            child[v].append(u)
        
        def dfs(node: int):
            nonlocal mask, d
            mask &= ~(1 << node)
            first = second = 0
            for p in child[node]:
                if mask & (1 << p):
                    mask &= ~(1 << p)
                    tmp = 1 + dfs(p)
                    if tmp > first:
                        first, second = tmp, first
                    elif tmp > second:
                        second = tmp
            d = max(d, first + second)
            return first

        res = [0] * (n - 1)
        for i in range(1, 1 << n):
            if i & (i - 1) == 0: continue
            mask, d = i, 0
            root = mask.bit_length() - 1
            dfs(root)
            if mask == 0:
                res[d - 1] += 1
        return res
```







---


# 常用数据结构

## 一、前缀和

### 原理

- 把$nums$记作$a$

$$
\begin{align}
s[0] &= 0 \\
s[1] &= a[0] \\
s[2] &= a[0]+a[1] \\
\vdots \\
s[n] &= \sum_{i=0}^{n-1}a[i] \\
\Rightarrow \sum_{i=left}^{right}a[i]&= s[right+1]-s[left] \\
\end{align}
$$

### 基础

#### 2438 二的幂数组中查询范围乘积

> 给定整数n，其由2的幂组成构造出一个数组。比如：15->[1，2，4，8]。queries查询这个数组中一段子数组的乘积，结果对$10^9+7$取模

- 如何构造这个数组，**二的幂要联想到位运算**
- 子数组的乘积，这里想到前缀和的思路，但是不能做前缀积（取模）
- 可以对幂次进行前缀和

```python
MOD = 10 ** 9 + 7
class Solution:
    def productQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        order = []
        for i in range(32):
            if (n >> i) & 1 == 1:
                order.append(i)
        preSum = [0] + list(accumulate(order))
        res = [0] * len(queries)
        for (i, [l, r]) in enumerate(queries):
            res[i] = (2 ** (preSum[r + 1] - preSum[l])) % MOD
        return res
```

### 前缀和+哈希表

#### 2588 统计美丽子数组数目

#### 3026 最大好子数组和

> 给你一个长度为 `n` 的数组 `nums` 和一个 **正** 整数 `k` 。
>
> 如果 `nums` 的一个子数组中，如果子数组 `nums[i..j]` 满足 `|nums[i] - nums[j]| == k` ，那么它是一个好子数组。
>
> 请你返回 `nums` 中 **好** 子数组的 **最大** 和，如果没有好子数组，返回 `0` 。

- 维护所有出现的数的下标随后遍历会tle
- 我们维护每个数的最小前缀和
- 更灵活的，我们还可以动态的维护前缀和

```python
class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        preSum = [0] + list(accumulate(nums))
        times = defaultdict(lambda: float('inf'))
        res = -inf
        for i in range(n):
            times[nums[i]] = min(times[nums[i]], preSum[i])
            def count(c):
                if c in times:
                    nonlocal res
                    res = max(res, preSum[i + 1] - times[c])
            count(nums[i] + k)
            count(nums[i] - k)
        return res if res != -inf else 0
```

#### 1546 和为目标值且不重叠的非空子数组的最大数目

> 给你一个数组 `nums` 和一个整数 `target` 。
>
> 请你返回 **非空不重叠** 子数组的最大数目，且每个子数组中数字和都为 `target` 。

- 一开始的做法是**前缀和+哈希表+dp**，时间复杂度是$O(n)$，但系数比较大，占空间也多
- 这题还有贪心的性质，我们从左往右遍历，如果存在以当前下标为右节点的子数组，那么结果加1
  - 证明：当前子数组满足结果且能提供最左的端点给下一个子数组

```python
class Solution:
    def maxNonOverlapping(self, nums: List[int], target: int) -> int:
        n = len(nums)
        preSum = [0] * (n + 1)
        times = defaultdict(int)
        dp = [0] * (n + 1)
        for i in range(n):
            times[preSum[i]] = i - 1
            preSum[i + 1] = preSum[i] + nums[i]
            if preSum[i + 1] - target in times:
                dp[i] = max(dp[i - 1], dp[times[preSum[i + 1] - target]] + 1)
            else:
                dp[i] = dp[i - 1]
        return dp[-2]
---
class Solution:
    def maxNonOverlapping(self, nums: List[int], target: int) -> int:
        i, n = 0, len(nums)
        preSum = res = 0
        times = set()
        while i < n:
            times.add(preSum)
            preSum += nums[i]
            if preSum - target in times:
                res += 1
                times = set()
            i += 1
        return res           
```

#### 1590 使数组和能被P整除

> 给你一个正整数数组 `nums`，请你移除 **最短** 子数组（可以为 **空**），使得剩余元素的 **和** 能被 `p` 整除。 **不允许** 将整个数组都移除。
>
> 请你返回你需要移除的最短子数组的长度，如果无法满足题目要求，返回 `-1` 。

- 设整个数组对$p$的模为$x$，那么我们需要找到子数组满足对$p$的模也为$x$
- 假设当前前缀和$preSum$对$p$的模为$y$，那么我们希望前面曾经出现过对$p$的模为$(y-x+p)\mod p$的前缀和，这样这个子数组就满足模为$x$
  - python保证取模结果为正所以不用额外$+p$

```python
class Solution:
    def minSubarray(self, nums: List[int], p: int) -> int:
        x = sum(nums) % p
        if x == 0:
            return 0
        remainder = {}
        res = n = len(nums)
        preSum = 0
        for i in range(n):
            remainder[preSum % p] = i - 1
            preSum += nums[i]
            j = remainder.get((preSum % p - x) % p, -n)
            res = min(res, i - j)
        return -1 if res >= n else res
```

## 二、差分数组

- 当遇到对**连续子数组**进行**同一**操作时可以考虑，除此之外连续子数组也要想到**滑动窗口**。
- [灵神的题单](https://leetcode.cn/circle/discuss/FfMCgb/)

$$
d(i) = \left\{
\begin{aligned}
a[0], \quad&i = 0 \\
a[i] - a[i-1], \quad&i > 0
\end{aligned}
\right.
$$

- 性质1: 累加可得原数组a
- 性质2:
  - 将$a[i], a[i+1], ...a[j]$加上x
  - 等价于$a[i]$加x，$a[j+1]$减x
    - 如果$j + 1 > n$，则不用操作

## 三、栈

### 单调栈

- 单调栈用于处理寻找下一个比其更小或更大的元素

#### 739 每日温度
- 最经典的单调栈
```cpp
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        stack<int> sk{};
        int n = temperatures.size();
        vector<int> res(n);
        for (int i = 0; i < n; i++) {
            while (!sk.empty() && temperatures[i] > temperatures[sk.top()]) {
                res[sk.top()] = i - sk.top();
                sk.pop();
            }
            sk.push(i);
        }
        return res;
    }
};
```

#### 84 柱状图中的最大矩形
- 85题中的单调栈更加容易理解，用单调栈计算两个数组分别记录当前坐标上下的比当前高度低的坐标。其中为了让每个元素都有值，使用了与模板不同的写法。
- 回到这道题其实也不是那么难理解，关键是前后要加一个0.
```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> sk;
        heights.insert(heights.begin(), 0);
        heights.push_back(0);
        sk.push(0);
        int res = 0, n = heights.size();
        for (int i = 1; i < n; i++) {
            while (!sk.empty() && heights[i] < heights[sk.top()]) {
                int mid = sk.top();
                sk.pop();
                if (!sk.empty()) {
                    int left = sk.top();
                    res = max(res, heights[mid] * (i - left - 1));
                }
            }
            sk.push(i);
        }
        return res;
    }
};
```

#### 85 最大矩形
```cpp
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        int i, j, res = 0;
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    dp[i][j] = (j > 0 ? dp[i][j-1] + 1: 1);
                }
            }
        }

        for (j = 0; j < n; j++) {
            stack<int> st;
            vector<int> up(m), down(m);
            for (int i = 0; i < m; i++) {
                while (!st.empty() && dp[i][j] <= dp[st.top()][j]) {
                    st.pop();
                }
                up[i] = st.empty() ? -1 : st.top();
                st.push(i);
            }
            st = stack<int>();
            for (int i = m - 1; i >= 0; i--) {
                while (!st.empty() && dp[i][j] <= dp[st.top()][j]) {
                    st.pop();
                }
                down[i] = st.empty() ? m : st.top();
                st.push(i);
            }
            for (int i = 0; i < m; i++) {
                res = max(res, dp[i][j] * (down[i] - up[i] - 1));
            }
        }
        return res;
    }
};
```



## 四、队列

### 单调队列

- 与单调栈思路相似

#### 239 滑动窗口最大值

> 给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。
>
> 返回 *滑动窗口中的最大值* 。

- 在一个窗口中，当我们遇到一个比单调队列尾部更大的数时，我们将单调队列尾部的数移除。这是因为后续这个尾部的数不会在是任何窗口的最大值了。
- 于是我们维护了一个单调递减的单调队列。

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = deque()
        res = []
        for i in range(k):
            while dq and nums[dq[-1]]<= nums[i]:
                dq.pop()
            dq.append(i)
        res.append(nums[dq[0]])
        n = len(nums)
        for i in range(k, n):
            if i - dq[0] + 1 > k:
                dq.popleft()
            while dq and nums[dq[-1]] <= nums[i]:
                dq.pop()
            dq.append(i)
            res.append(nums[dq[0]])
        return res
```





## 六、字典树Trie

### 基础

```python
class Trie:

    class TrieNode:
        def __init__(self, isWord: bool):
            self.isWord = isWord
            self.children = {}
        
        def goDown(self, c) -> Optional["TrieNode"]:
            if c not in self.children:
                return None
            return self.children[c]

        def goDownandCreate(self, c, isWord) -> "TrieNode":
            if c not in self.children:
                self.children[c] = Trie.TrieNode(isWord)
            if isWord:
                self.children[c].isWord = isWord
            return self.children[c]

    def __init__(self):
        self.root = self.TrieNode(False)

    def insert(self, word: str) -> None:
        n = len(word)
        node = self.root
        for i in range(n - 1):
            node = node.goDownandCreate(word[i], False)
        node = node.goDownandCreate(word[n - 1], True)

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            node = node.goDown(c)
            if node is None:
                return False  
        return node.isWord

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            node = node.goDown(c)
            if node is None:
                return False
        return True

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

### 进阶

- 基本就是`TrieNode`中存储的特征变了

#### 3093 最长公共后缀查询（2118）

> 给你两个字符串数组 `wordsContainer` 和 `wordsQuery` 。
>
> 对于每个 `wordsQuery[i]` ，你需要从 `wordsContainer` 中找到一个与 `wordsQuery[i]` 有 **最长公共后缀** 的字符串。如果 `wordsContainer` 中有两个或者更多字符串有最长公共后缀，那么答案为长度 **最短** 的。如果有超过两个字符串有 **相同** 最短长度，那么答案为它们在 `wordsContainer` 中出现 **更早** 的一个。
>
> 请你返回一个整数数组 `ans` ，其中 `ans[i]`是 `wordsContainer`中与 `wordsQuery[i]` 有 **最长公共后缀** 字符串的下标。

- 出现更早只需要正向遍历即可满足，我们在`TrieNode`中记录长度和索引

```python
class TrieNode:
    def __init__(self):
        self.word_i = inf
        # len(word)
        self.word_l = inf
        self.children = {}

    def goDownandCreate(self, c: str, word_i: int, word_l: int) -> "TrieNode":
        if c not in self.children:
            self.children[c] = TrieNode()
        if word_l < self.children[c].word_l:
            self.children[c].word_i = word_i
            self.children[c].word_l = word_l
        return self.children[c]

    def goDown(self, c: str) -> Optional["TrieNode"]:
        if c not in self.children:
            return None
        return self.children[c]

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str, word_i: int) -> None:
        node = self.root
        n = len(word)
        if n < node.word_l:
            node.word_l = n
            node.word_i = word_i
        # 最长后缀, 所以反向插入
        for i in range(n - 1, -1, -1):
            node = node.goDownandCreate(word[i], word_i, n)
    
    def searchSuffix(self, word: str) -> int:
        node = self.root
        for s in reversed(word):
            tmp = node.goDown(s)
            if tmp is None:
                return node.word_i
            node = tmp
        return node.word_i

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        trie = Trie()
        for i, s in enumerate(wordsContainer):
            trie.insert(s, i)
        ans = [0] * (len(wordsQuery))
        for i, query in enumerate(wordsQuery):
            ans[i] = trie.searchSuffix(query)
        return ans
```



## 七、并查集

### 模板

```python
class UFSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.ranks = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        a, b = self.find(x), self.find(y)
        if a != b:
            if self.ranks[a] < self.ranks[b]:
                self.parent[a] = b
            elif self.ranks[a] > self.ranks[b]:
                self.parent[b] = a
            else:
                self.parent[b] = a
                self.ranks[a] += 1
```



### 基础

#### 990 等式方程的可满足性 

> - `a==b` `b!=a` 不满足
> - `a==b` `b==c` `a!=c` 不满足

```python
class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        us = UFSet(26)
        for e in equations:
            if e[1] == '=':
                us.union(ord(e[0]) - ord('a'), ord(e[3]) - ord('a'))
        for e in equations:
            if e[1] == '!':
                if us.find(ord(e[0]) - ord('a')) == us.find(ord(e[3]) - ord('a')):
                    return False
        return True
```



---

[位运算题单](https://leetcode.cn/circle/discuss/dHn9Vk/)

[分享｜从集合论到位运算，常见位运算技巧分类总结！ - 力扣（LeetCode）](https://leetcode.cn/circle/discuss/CaOJ45/)

## 位运算

#### 相关技巧

- $x \oplus y \oplus y = 0$
  - 记$sumXor(x) = 0 \oplus 1 \oplus 2 \oplus...\oplus x$
  - 性质1： $s\oplus(s+1)...\oplus n = sumXor(s-1) \oplus sumXor(n)$
- $4i\oplus(4i+1)\oplus(4i+2)\oplus(4i+3)=0$
  - 对于任意2的倍数都成立

#### 191 计算1的个数

- 比较容易想到的是每次清零最低1位的1，看多少次变为0. 复杂度 $O(logn)$​
  - `n & (n - 1)`
- 最高效的是使用标准库，标准库利用分组的方法能更快.

#### 2588 统计美丽子数组数目 (1697)

- 曾经没做出来的周赛第三题，现在已经可以完成了
- 经过翻译，就是在数组求异或和位0的连续子数组的个数
- 前缀和（因为上面的性质1）
  - 我们希望找到一对数，其异或和为0。这说明这两个数相同，我们用哈希表记录之，随后$\binom{n}{2}$即可。

```cpp
class Solution {
public:
    long long beautifulSubarrays(vector<int>& nums) {
        int n = nums.size();
        vector<int> sumXor(n + 1);
        sumXor[0] = nums[0];
        long long res = nums[0] == 0;
        for (int i = 1; i < n; i++) {
            sumXor[i] = nums[i] ^ sumXor[i - 1];
            if (sumXor[i] == 0)
                res++;
        }
        unordered_map<int, int> memo{};
        for (int i = 0; i < n; i++)
            if (memo.count(sumXor[i]))
                memo[sumXor[i]]++;
            else
                memo.emplace(sumXor[i], 1);
        for (auto [_, y]: memo) {
            res += (long long)y * (y - 1) /2;
        }
        return res;
    }
};
```



# Leetcode周赛

## 390 周赛（虚拟竞赛）

#### T3. 最高频率的ID

> 你需要在一个集合里动态记录 ID 的出现频率。给你两个长度都为 `n` 的整数数组 `nums` 和 `freq` ，`nums` 中每一个元素表示一个 ID ，对应的 `freq` 中的元素表示这个 ID 在集合中此次操作后需要增加或者减少的数目。
>
> - **增加 ID 的数目：**如果 `freq[i]` 是正数，那么 `freq[i]` 个 ID 为 `nums[i]` 的元素在第 `i` 步操作后会添加到集合中。
> - **减少 ID 的数目：**如果 `freq[i]` 是负数，那么 `-freq[i]` 个 ID 为 `nums[i]` 的元素在第 `i` 步操作后会从集合中删除。
>
> 请你返回一个长度为 `n` 的数组 `ans` ，其中 `ans[i]` 表示第 `i` 步操作后出现频率最高的 ID **数目** ，如果在某次操作后集合为空，那么 `ans[i]` 为 0 。

- 在每次取最大值的时候，我们需要遍历`count`这会TLE
- 使用`multiset`维护所有的次数
  - `erase`元素会删除所有这个元素，而我们希望的是删除其中一个，使用迭代器
  - `find`的时候注意判断是否为`end`

```c++
class Solution {
public:
    vector<long long> mostFrequentIDs(vector<int>& nums, vector<int>& freq) {
        int n = nums.size();
        vector<long long> res(n);
        unordered_map<int, long long> times{};
        multiset<long long> p;
        for (int i = 0; i < n; i++) {
            times[nums[i]] += freq[i];
            p.insert(times[nums[i]]);
            auto it = p.find(times[nums[i]] - freq[i]);
            if (it != p.end())
                p.erase(it);
            res[i] = *p.rbegin();
        }
        return res;
    }
};
```



## 398周赛（1854 / 3605）

- 总结：掉大分，第二题一直爆内存，第三题手残用了`static`错了8次才找到bug，第四题不难但理解还是不到位，做错了。

#### T2. 特殊数组II

> 如果数组的每一对相邻元素都是两个奇偶性不同的数字，则该数组被认为是一个 **特殊数组** 。
>
> 有一个整数数组 `nums` 和一个二维整数矩阵 `queries`，对于 `queries[i] = [fromi, toi]`，检查子数组 `nums[fromi..toi]` 是不是一个 **特殊数组** 。
>
> 返回布尔数组 `answer`，如果 `nums[fromi..toi]` 是特殊数组，则 `answer[i]` 为 `true` ，否则，`answer[i]` 为 `false` 。

- 爆内存做法

```c++
class Solution {
public:
    vector<bool> isArraySpecial(vector<int>& nums, vector<vector<int>>& queries) {
        int n = nums.size();
        vector<vector<bool>> dp(n, vector<bool>(n));
        for (int i = 0; i < n; i++)
            dp[i][i] = true;
        for (int i = 0; i < n; i++) {
            for (int j = i - 1; j >= 0; j--) {
                bool flag1 = nums[j] % 2 == 0;
                bool flag2 = nums[j + 1] % 2 == 0;
                if ((flag1 ^ flag2) == 1)
                    dp[j][i] = dp[j + 1][i];
                else
                    dp[j][i] = false;
            }
        }
        int m = queries.size();
        vector<bool> res(m);
        for (int i = 0; i < m; i++)
            res[i] = dp[queries[i][0]][queries[i][1]]; 
        return res;
    }
};
```

- 事实上，这种区间式的查询很容易想到前缀和。然而考试的时候没有细想，实际上只要创建一个新数组，如果相邻奇偶性一样则赋值为0。那么我们区间和为0的数组就是**特殊数组**了

```c++
class Solution {
public:
    vector<bool> isArraySpecial(vector<int>& nums, vector<vector<int>>& queries) {
        int n = nums.size();
        vector<int> preSum(n);

        for (int i = 1; i < n; i++) {
            int tmp = !((nums[i] & 1) ^ (nums[i - 1] & 1));
            preSum[i] = preSum[i - 1] + tmp;
        }

        int m = queries.size();
        vector<bool> res(m); 
        for (int i = 0; i < m; i++) {
            res[i] = (preSum[queries[i][1]] - preSum[queries[i][0]] == 0);
        }
        return res;
    }
};
```

#### T4. 到达第k级台阶的方案数

> 给你有一个 **非负** 整数 `k` 。有一个无限长度的台阶，**最低** 一层编号为 0 。
>
> 虎老师有一个整数 `jump` ，一开始值为 0 。虎老师从台阶 1 开始，虎老师可以使用 **任意** 次操作，目标是到达第 `k` 级台阶。假设虎老师位于台阶 `i` ，一次 **操作** 中，虎老师可以：
>
> - 向下走一级到 `i - 1` ，但该操作 **不能** 连续使用，如果在台阶第 0 级也不能使用。
> - 向上走到台阶 `i + 2^jump` 处，然后 `jump` 变为 `jump + 1` 。
>
> 请你返回虎老师到达台阶 `k` 处的总方案数。
>
> **注意** ，虎老师可能到达台阶 `k` 处后，通过一些操作重新回到台阶 `k` 处，这视为不同的方案。

- 当时写法，因为太着急没有细想，这里dfs根本没有返回值，cache是完全错误的，这也体现了我对其的理解不深，考场不够冷静。

```python
class Solution:
    def waysToReachStair(self, k: int) -> int:
        ans = 0
        @cache
        def dfs(i, flag, jump):
            if i > k + 1:
                return
            if i == k:
                nonlocal ans
                ans += 1
            if flag and i > 0:
                dfs(i - 1, False, jump)
            dfs(i + 2 ** jump, True, jump + 1)
        dfs(1, True, 0)
        return ans
```

- 我们约定$dfs(i, ...)$表示从$i$到$k$的方案数

```python
class Solution:
    def waysToReachStair(self, k: int) -> int:
        @cache
        def dfs(i, flag, jump):
            if i > k + 1:
                return 0
            res = 0
            if i == k:
                res = 1
            if flag and i > 0:
                res += dfs(i - 1, False, jump)
            res += dfs(i + 2 ** jump, True, jump + 1)
            return res
        return dfs(1, True, 0)
```



























