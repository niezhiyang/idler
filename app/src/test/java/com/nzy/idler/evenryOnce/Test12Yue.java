package com.nzy.idler.evenryOnce;


import com.nzy.idler.NodeUtils;
import com.nzy.idler.model.ListNode;
import com.nzy.idler.model.TreeNode;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

/**
 * @author niezhiyang
 * since 2024/11/26
 */
public class Test12Yue {

    @Test
    public void reverseKGroupTest() {
        ListNode node1 = new ListNode(1);
        ListNode node2 = new ListNode(2);
        node1.next = node2;
        ListNode node3 = new ListNode(3);
        node2.next = node3;
        ListNode node4 = new ListNode(4);
        node3.next = node4;
        ListNode node5 = new ListNode(5);
        node4.next = node5;
        ListNode node6 = new ListNode(6);
        node5.next = node6;
        ListNode node7 = new ListNode(7);
        node6.next = node7;
        reverseKGroup(node1, 3);

    }

    public ListNode reverseKGroup(ListNode head, int k) {
        // 1->2->3->4->5->6
        if (head == null) {
            return null;
        }
        ListNode end = head;
        for (int i = 0; i < k - 1; i++) {
            end = end.next;
            if (end == null) {
                return head;
            }
        }
        // end 假如现在 3-4-5-6
        System.out.println("end : " + NodeUtils.printList(end));
        // nextListNode = 4->5->6
        ListNode nextListNode = end.next;
        // 3->2->1
        ListNode swap = reverseListNode(head, end);
        System.out.println("temp : " + NodeUtils.printList(swap));
        // temp 是 4-5-6 后面的反转
        ListNode temp = reverseKGroup(nextListNode, k);
        head.next = temp;
        return swap;

    }

    private ListNode reverseListNode(ListNode start, ListNode end) {
        ListNode tmp = null;
        ListNode temp = start;
        while (tmp != end) {
            ListNode next = temp.next;
            temp.next = tmp;
            tmp = temp;
            temp = next;
        }
        return tmp;
    }


    @Test
    public void testthreeSum() {
        threeSum(new int[]{-1, 0, 1, 2, -1, -4});
    }

    public List<List<Integer>> threeSum(int[] nums) {
        return null;
    }

    /**
     * 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     *
     * @param nums
     * @return 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
     * dp[0] = -2
     * dp[1] = 1
     * dp[2] =
     * 输出：6
     * 解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
     * <p>
     * dp[i] = Math.max(dp[i-1]+nums[i],nums[i]) dp[i]带表示前i个元素中的最大和
     */
    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        int result = dp[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
            result = Math.max(dp[i], result);
        }
        return result;
    }

    public int maxSubArray2(int[] nums) {
        int pre = nums[0];
        int cur = nums[0];
        int result = pre;
        for (int i = 1; i < nums.length; i++) {
            cur = Math.max(pre + nums[i], nums[i]);
            result = Math.max(cur, result);
            // 重新复制
            pre = cur;


        }
        return result;
    }

    /**
     * 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
     *
     * @return
     */

    @Test
    public void testMerge() {
        ListNode node111 = new ListNode(1);
        ListNode node112 = new ListNode(3);
        ListNode node113 = new ListNode(5);
        ListNode node114 = new ListNode(7);
        node111.next = node112;
        node112.next = node113;
        node113.next = node114;


        ListNode node21 = new ListNode(2);
        ListNode node22 = new ListNode(4);
        ListNode node23 = new ListNode(6);
        ListNode node24 = new ListNode(8);
        node21.next = node22;
        node22.next = node23;
        node23.next = node24;

        System.out.println("合并有序链表：" + NodeUtils.printList(mergeTwoLists(node21, node111)));
    }

    public ListNode mergeTwoLists(ListNode head1, ListNode head2) {
        // 穿针引线
        ListNode result = new ListNode(-1);
        ListNode temp = result;
        while (head1 != null && head2 != null) {
            if (head1.val > head2.val) {
                temp.next = head2;
                head2 = head2.next;
            } else {
                temp.next = head1;
                head1 = head1.next;
            }
            temp = temp.next;
        }
        if (head1 != null) {
            temp.next = head1;
        } else {
            temp.next = head2;
        }

        return result.next;

    }

    @Test

    public void testLongest() {
//        s = "babad"
//        输出："bab"
//        解释："aba" 同样是符合题意的答案。
        System.out.println(longestPalindrome("cbbd"));

    }

    /**
     * 给你一个字符串 s，找到 s 中最长的
     * 回文
     * <p>
     * 子串
     * 。
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        String result = "";
        for (int i = 0; i < s.length(); i++) {
            String indexI = getLengthStr(s, i, i);
            String indexNext = getLengthStr(s, i, i + 1);
            if (indexI.length() > result.length()) {
                result = indexI;
            }
            if (indexNext.length() > result.length()) {
                result = indexNext;
            }
        }
        return result;
    }

    private String getLengthStr(String s, int left, int right) {
        String result = "";
        while (left >= 0 && right < s.length()) {
            if (s.charAt(left) == s.charAt(right)) {
                result = s.substring(left, right + 1);
                System.out.println(result + "-....--" + left + "---" + right);
                left--;
                right++;
            } else {
                break;
            }
        }
        System.out.println(result + "---" + left + "---" + right);
        return result;

    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new LinkedList<>();
        LinkedList<TreeNode> list = new LinkedList<>();
        if (root == null) {
            return result;
        }
        list.add(root);
        while (!list.isEmpty()) {
            int length = list.size();
            List<Integer> temp = new ArrayList<>();
            for (int i = 0; i < length; i++) {
                TreeNode node = list.removeFirst();
                temp.add(node.val);
                if (node.left != null) {
                    list.addLast(node.left);
                }
                if (node.right != null) {
                    list.addLast(node.right);
                }
            }
            result.add(temp);
        }
        return result;
    }

    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int jValue = target - nums[i];
            if (map.containsKey(jValue)) {
                return new int[]{map.get(jValue), i};
            } else {
                map.put(nums[i], i);
            }
        }
        return new int[]{};
    }

    /*    5
       4
                3
              2
            1
     */
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (right > left) {
            int mid = (right + left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                if (nums[mid] < nums[right]) {
                    // 证明 mid 到 right 是 递增
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                if (nums[mid] > nums[left]) {
                    // 证明 left 到 mid 是 递增
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

        }
        return -1;
    }

    public int search4(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (right >= left) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                System.out.println("找到: " + mid);
                return mid;
            }
            if (nums[mid] >= nums[left]) {
                // left -> mid 是递增了
                if (target >= nums[left] && target < nums[mid]) {
                    // 在这个递增之间
                    right = mid - 1;
                } else {
                    // 不在这个区间
                    left = mid + 1;
                }
            } else {
                //  mid -> right 是递增了
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }

        }
        return -1;
    }

    public int search2(int[] nums, int target) {
        int length = nums.length;
        if (length == 0) {
            return -1;
        }
        if (length == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int left = 0, right = length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            // 若果left right 就是的话直接返回
            if (nums[left] == target) {
                return left;
            }
            if (nums[right] == target) {
                return right;
            }
            // 左边是升序
            // [5,6,1],[2,3,4]
            if (nums[0] < nums[mid]) {
                if (nums[0] < target && target < nums[mid]) {
                    // 目标值在左侧
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                // 右边是升序
                if (nums[mid] < target && target < nums[length - 1]) {
                    // 目标值在右侧
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    /**
     * 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * <p>
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * <p>
     * 此外，你可以假设该网格的四条边均被水包围。
     *
     * @param grid
     * @return
     */
    @Test
    public void testnumisland() {
        char[][] demo = new char[][]{
                new char[]{'1', '1', '1', '1', '0'},
                new char[]{'1', '1', '0', '1', '0'},
                new char[]{'1', '1', '0', '0', '0'},
                new char[]{'0', '0', '0', '0', '0'}};
        System.out.println(numIslands(demo));
    }

    public int numIslands(char[][] grid) {
        int result = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    // 开始遍历 上下左右，设置成0
                    dfs(i, j, grid);
                    NodeUtils.printTwoArray(grid);
                    System.out.println("---------");
                    result++;
                }
            }
        }
        return result;

    }

    private void dfs(int i, int j, char[][] grid) {
        if (i >= 0 && i < grid.length && j >= 0 && j < grid[0].length) {
            if (grid[i][j] == '1') {
                grid[i][j] = '0';
                dfs(i + 1, j, grid);
                dfs(i - 1, j, grid);
                dfs(i, j - 1, grid);
                dfs(i, j + 1, grid);
            }
        }

    }

    /**
     * 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
     * <p>
     * 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
     * <p>
     * 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
     *
     * @param prices
     * @return
     */
    @Test
    public void testMaxpro() {
        System.out.println(maxProfit(new int[]{7, 1, 5, 3, 6, 4}));
    }

    public int maxProfit(int[] prices) {
        int minDayValue = prices[0];
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > minDayValue) {
                result = Math.max(prices[i] - minDayValue, result);
            } else {
                minDayValue = prices[i];
            }
        }
        return result;

    }

    /**
     * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
     * 示例 1：
     * <p>
     * 输入：nums = [1,2,3]
     * 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
     *
     * @param nums
     * @return
     */

    @Test
    public void textPermute() {
        List<List<Integer>> result = permute(new int[]{1, 2, 3});
        for (int i = 0; i < result.size(); i++) {

            System.out.println(Arrays.toString(result.get(i).toArray()));
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        dfsList(nums, result, new ArrayList<Integer>());
        return result;
    }

    private void dfsList(int[] nums, List<List<Integer>> result, ArrayList<Integer> temp) {

        if (temp.size() == nums.length) {
            result.add(new ArrayList<>(temp));
        } else {
            for (int j = 0; j < nums.length; j++) {
//                if (!temp.contains(nums[j])) {
//                    ArrayList<Integer> newTemp = new ArrayList<>(temp);
//                    newTemp.add(nums[j]);
//                    dfsList(nums, result, newTemp);
//                }
                if (!temp.contains(nums[j])) {
                    temp.add(nums[j]);
                    dfsList(nums, result, temp);
                    temp.remove(temp.size() - 1);
                }
            }

        }
    }

    /**
     * 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
     * <p>
     * 有效字符串需满足：
     * <p>
     * 左括号必须用相同类型的右括号闭合。
     * 左括号必须以正确的顺序闭合。
     * 每个右括号都有一个对应的相同类型的左括号。
     *
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } else {
                if (stack.isEmpty()) {
                    return false;
                }
                char pop = stack.pop();
                switch (c) {
                    case ')':
                        if (pop != '(') {
                            return false;
                        }
                        break;
                    case '}':
                        if (pop != '{') {
                            return false;
                        }
                        break;
                    case ']':
                        if (pop != '[') {
                            return false;
                        }
                        break;
                }
            }
        }
        return stack.isEmpty();
    }

    /**
     * 给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。
     * <p>
     * 请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。
     * <p>
     * 注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
     * 输出：[1,2,2,3,5,6]
     * 解释：需要合并 [1,2,3] 和 [2,5,6] 。
     * 合并结果是 [1,2,2,3,5,6] ，其中斜体加粗标注的为 nums1 中的元素。
     *
     * @param nums1
     * @param m
     * @param nums2
     * @param n
     */
    @Test
    public void textMerge() {
        int[] num1 = new int[]{1, 2, 3, 0, 0, 0};
        merge(num1, 3, new int[]{2, 5, 6}, 3);

//        int[] num1 = new int[]{2, 5, 6, 0, 0, 0};
//        merge(num1, 3, new int[]{1, 1, 1}, 3);
//        int[] num1 = new int[]{0};
//        merge(num1, 0, new int[]{1}, 1);
        System.out.println(Arrays.toString(num1));
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        // 从后往前排
        int indexM = m - 1;
        int indexN = n - 1;
        while (indexM > -1 || indexN > -1) {
            if (indexN == -1) {
                indexM--;
            } else if (indexM == -1) {
                nums1[indexN] = nums2[indexN];
                indexN--;
            } else if (nums1[indexM] > nums2[indexN]) {
                nums1[indexM + indexN + 1] = nums1[indexM];
                indexM--;
            } else {
                nums1[indexM + indexN + 1] = nums2[indexN];
                indexN--;
            }
        }

    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        // Z 字型
        LinkedList<TreeNode> queue = new LinkedList<>();
        if (root != null) {
            queue.add(root);
        }
        List<List<Integer>> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            LinkedList<Integer> list = new LinkedList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode first = queue.removeFirst();
                if (result.size() % 2 == 0) {
                    list.addLast(first.val);
                } else {
                    list.addFirst(first.val);
                }
                if (first.left != null) {
                    queue.add(first.left);
                }
                if (first.right != null) {
                    queue.add(first.right);
                }
            }
            result.add(list);

        }
        return result;
    }

    /**
     * 1
     * 2         3
     * 4    5    6     7
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (p == root) {
            return p;
        }
        if (q == root) {
            return q;
        }


        return null;

    }

    public boolean hasCycle(ListNode head) {
        if (head == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;

        while (fast != null && fast.next != null) {
            if (slow == fast) {
                return true;
            } else {
                slow = slow.next;
                fast = fast.next.next;
            }
        }
        return false;

    }


    @Test
    public void testreverseBetween() {
        ListNode node1 = new ListNode(1);
        ListNode node2 = new ListNode(2);
        node1.next = node2;
        ListNode node3 = new ListNode(3);
        node2.next = node3;
        ListNode node4 = new ListNode(4);
        node3.next = node4;
        ListNode node5 = new ListNode(5);
        node4.next = node5;
        ListNode node6 = new ListNode(6);
        node5.next = node6;
        ListNode node7 = new ListNode(7);
        node6.next = node7;
        ListNode result = reverseBetween(node1, 3, 4);
        System.out.println("结果是：" + NodeUtils.printList(result));

    }

    public ListNode reverseBetween(ListNode head, int left, int right) {
//        ListNode preHead = new ListNode(0);
//        preHead.next = head;
        ListNode leftNode = head;
        ListNode rightNode = head;
        ListNode leftNodePre = null;
        // 如果 left 和 right 从1开始，那么这里到i也从1开始就行了
        for (int i = 1; i < right; i++) {
            if (i < left) {
                leftNodePre = leftNode;
                leftNode = leftNode.next;

            }
            rightNode = rightNode.next;

        }


        // 避免影响到原来的值
        ListNode rightNext = rightNode.next;
        ListNode leftTemp = leftNode;
        // 1->2->3->4-5-6  , 3,4
        // leftPre = 1->2
        // left = 3->4-5-6
        // right = 4-5-6
        // rightNext = 5-6
        ListNode revert = revert(leftTemp, rightNext);
        // revert 是 4->3
        System.out.println("revert：" + NodeUtils.printList(revert));
        leftNode.next = rightNext;
        // 变了 leftNode = > 4-3-5-6
        if (leftNodePre != null) {
            leftNodePre.next = revert;
        } else {
            // 也就是从left = 1
            return revert;
        }

        return head;


    }

    private ListNode revert(ListNode leftNode, ListNode rightNode) {
        ListNode pre = null;
        ListNode cur = leftNode;
        while (cur != rightNode) {
            ListNode temp = cur.next;
            cur.next = pre;

            pre = cur;
            cur = temp;
        }
        return pre;
    }

    public ListNode reverseBetween1(ListNode head, int left, int right) {
        // （1）初始化指针
        ListNode preHead = new ListNode(0);
        preHead.next = head;
        // [1,2,3,4,5] , left = 2, right = 4  ==》 [1,4,3,2,5]
        // 那么g是 1 ,p是 2

        // g是反转的前一个
        // p是要反转的第一个
        // （2）将 p 后面的元素删除,然后添加到 g 的后面。也即头插法。
        // 根据 m 和 n 重复步骤（2）
        ListNode preCur = preHead;
        ListNode cur = preHead.next;

        // 将指针移到相应的位置
        for (int step = 0; step < left - 1; step++) {
            preCur = preCur.next;
            cur = cur.next;
        }

        // 头插法插入节点,也就是执行 right - left 次数
        for (int i = 0; i < right - left; i++) {
            //要移除的 比如测试 removed 是 3->4->5
            ListNode removed = cur.next;
            // 删除 removed 节点
            cur.next = cur.next.next;
            // 再把这个节点插入到 g的后面
            removed.next = preCur.next;
            preCur.next = removed;
        }

        return preHead.next;
    }

    public ListNode reverseBetween11(ListNode head, int left, int right) {
        // 1. 初始化一个虚拟头节点，它的 next 是 head。
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode leftNodePre = dummy;   // 操作指针
        ListNode rightNode = dummy;   // 操作指针

        // 2. 移动操作指针 left - 1 次，实际上是到了要反转位置的前一个位置
        for (int i = 0; i < left - 1; i++) {
            leftNodePre = leftNodePre.next;
        }
        // 3. 移动操作指针 right - 1 次，实际上是到了要反转位置的前一个位置
        for (int i = 0; i < right - 1; i++) {
            rightNode = rightNode.next;
        }

        // 3. 第三部分：正常的一维反转逻辑，但是 pre 节点不能到达 head 尾部了，而是到达反转区域的下一个节点的位置。比如要反转 [2,4]，那就移动到 5，方便原链表节点连接反转后的链表。
        ListNode pre = leftNodePre.next;
        ListNode tail = null;

        for (int i = 0; i < right - left + 1; i++) {    // 原版的是while(pre != null)
            ListNode t = pre.next;
            pre.next = tail;
            tail = pre;
            pre = t;
        }

        // 4. 第四部分：拼接上反转后的链表.
        // - cur 此时还在反转区域的前一个节点，用它进行连接即可。
        // - 此时pre 节点在反转区域的后面一个节点。tail 就在反转区域的最后一个节点
        // 拼接：cur.next.next，实际上是操作反转区域的第一个节点的 next 到 pre。然后是cur.next表示换成 tail节点，就拼好了。

        leftNodePre.next.next = pre;
        leftNodePre.next = tail;

        return dummy.next;
    }
}
