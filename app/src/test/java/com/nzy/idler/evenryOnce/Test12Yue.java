package com.nzy.idler.evenryOnce;


import com.nzy.idler.NodeUtils;
import com.nzy.idler.model.ListNode;
import com.nzy.idler.model.TreeNode;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
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
        System.out.println("leftPre:" + NodeUtils.printList(leftNodePre));
        // left = 3->4-5-6
        System.out.println("leftNode:" + NodeUtils.printList(leftNode));
        // right = 4-5-6
        System.out.println("rightNode:" + NodeUtils.printList(rightNode));
        ListNode revert = revert(leftTemp, rightNext);
        System.out.println("leftNode:" + NodeUtils.printList(leftNode));
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

    /**
     * [ 1, 2, 3]
     * [ 8, 9, 4]
     * [ 7, 6, 5]
     *
     * @param matrix
     * @return
     */

    @Test
    public void testSpira() {
        List<Integer> result = spiralOrder(new int[][]{
                new int[]{1, 2, 3},
                new int[]{8, 9, 4},
                new int[]{8, 9, 4},
                new int[]{7, 6, 5}
        });
        System.out.println(Arrays.toString(result.toArray()));
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        int m = matrix.length;
        if (m == 0) {
            return result;
        }
        int n = matrix[0].length;
        int top = 0;
        int bottom = matrix.length - 1;

        int left = 0;
        int right = matrix[0].length - 1;

        while (true) {
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            top++;
            if (bottom < top) {
                break;
            }

            for (int i = top; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            right--;
            if (right < left) {
                break;
            }


            for (int i = right; i >= left; i--) {
                result.add(matrix[bottom][i]);
            }
            bottom--;
            if (bottom < top) {
                break;
            }

            for (int i = bottom; i >= top; i--) {
                result.add(matrix[i][left]);
            }
            left++;
            if (right < left) {
                break;
            }
        }
        return result;
    }

    /**
     * 1 2  3
     * 2  3
     * <p>
     * 字符串相加
     */
    public String addStrings(String num1, String num2) {
        StringBuilder result = new StringBuilder();
        int index1 = num1.length() - 1;
        int index2 = num2.length() - 1;
        int ret = 0;
        while (index1 >= 0 || index2 >= 0) {
            char c1 = '0';
            if (index1 >= 0) {
                c1 = num1.charAt(index1);
            }
            char c2 = '0';
            if (index2 >= 0) {
                c2 = num2.charAt(index2);
            }
            int temp = (c1 - '0') + (c2 - '0') + ret;
            result.append(temp % 10);
            ret = temp / 10;
            index1--;
            index2--;
        }
        if (ret == 1) {
            result.append(ret);
        }
        return result.reverse().toString();
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int lengthA = getListNodeLength(headA);
        int lengthB = getListNodeLength(headB);
        if (lengthA > lengthB) {
            // 先走
            int step = lengthA - lengthB;
            for (int i = 0; i < step; i++) {
                headA = headA.next;
            }
        } else {
            // 先走
            int step = lengthB - lengthA;
            for (int i = 0; i < step; i++) {
                headB = headB.next;
            }
        }
        while (headA != null) {
            if (headA == headB) {
                return headA;
            }
            headA = headA.next;
            headB = headB.next;
        }
        return null;
    }

    private int getListNodeLength(ListNode headA) {
        ListNode temp = headA;
        int length = 0;
        while (temp != null) {
            temp = temp.next;
            length++;
        }
        return length;

    }

    /*给定一个单链表 L 的头节点 head ，单链表 L 表示为：

L0 → L1 → … → Ln - 1 → Ln
请将其重新排列后变为：

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。*/
    @Test
    public void testReorderList() {
        ListNode node1 = new ListNode(1);
        ListNode node2 = new ListNode(2);
        node1.next = node2;
        ListNode node3 = new ListNode(3);
        node2.next = node3;
        ListNode node4 = new ListNode(4);
        node3.next = node4;
        reorderList(node1);
    }

    public void reorderList(ListNode head) {
        // 先存放在list中
        LinkedList<ListNode> nodes = new LinkedList<>();
        ListNode temp = head;
        while (temp != null) {
            nodes.add(temp);
            temp = temp.next;
        }
        ListNode preLast = null;
        while (!nodes.isEmpty()) {
            // 先拿第一个
            ListNode first = nodes.removeFirst();
            // 把first的下一个设置未null
            first.next = null;
            if (preLast != null) {
                // 把上一个的last给到first
                preLast.next = first;
            }
            ListNode last = null;
            if (!nodes.isEmpty()) {
                // 再拿倒数
                last = nodes.removeLast();
                // 把last的下一个设置未null
                last.next = null;
                // 记录 last
                preLast = last;

            }
            // 把first的next 给到last
            first.next = last;

        }
    }

    /**
     * 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
     * 请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
     * <p>
     * 1 <= intervals.length <= 104
     * intervals[i].length == 2
     * 0 <= starti <= endi <= 104
     * <p>
     * 示例 1：
     * <p>
     * 输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
     * 输出：[[1,6],[8,10],[15,18]]
     * 解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
     *
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {
        // 先排序
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });
        ArrayList<int[]> list = new ArrayList<>();
        // 先记录left和right
        int right = intervals[0][1];
        int left = intervals[0][0];
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] > right) {
                // 如果当前大于了，那么就不能合并
                list.add(new int[]{left, right});

                // 然后再设置新值
                left = intervals[i][0];
                right = intervals[i][1];
            } else {
                // 如果当前的right 大于 right，那么right就能合并
                if (intervals[i][1] > right) {
                    right = intervals[i][1];
                }
            }
        }
        // 最后一个添加到集合中
        list.add(new int[]{left, right});
        int[][] result = new int[list.size()][2];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }
        return result;

    }

    /**
     * 1->2->3->4->5->6
     * |     |
     * 9<-8<-7
     * <p>
     * a+nb = (a+mb)*2
     * nb=a+2mb
     * (n-2m)b = a
     *
     * @param head
     * @return
     */
    // 循环连标入口，非循环返回null
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        boolean hasCycle = false;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                hasCycle = true;
                break;
            }
        }
        if (hasCycle) {
            ListNode temp = head;
            while (temp != fast) {
                temp = temp.next;
                fast = fast.next;
            }
            return fast;
        }

        return null;
    }

    /**
     * text1 = "abcde", text2 = "ace"
     * a b c d e
     * a 1 0 0 0 0
     * c 1 1 2 2 2
     * e 1 1 2 2 3
     *
     * @param text1
     * @param text2
     * @return
     */
    public int longestCommonSubsequence(String text1, String text2) {
        int length1 = text1.length();
        int length2 = text2.length();
        int[][] dp = new int[length1 + 1][length2 + 1];
        int result = 0;

        for (int i = 0; i < length1; i++) {
            for (int j = 0; j < length2; j++) {
                dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
                if (text1.charAt(i) == text2.charAt(j)) {
                    // 如果当前一样
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                    result = Math.max(dp[i + 1][j + 1], result);
                }
            }
        }
        return result;

    }

    /**
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode cur = head;
        ListNode fast = head;
        ListNode curPre = null;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        // 从头直接就移除
        if (fast == null) {
            return head.next;
        }

        while (fast != null) {
            curPre = cur;
            fast = fast.next;
            cur = cur.next;
        }
        curPre.next = cur.next;

        return head;

    }
    // 给定一个已排序的链表的头 head ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表 。


    /**
     * 输入：head = [1,2,3,3,4,4,5]
     * 输出：[1,2,5]
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return null;
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            if (cur.val == cur.next.val) {
                int val = cur.val;
                while (cur != null && cur.val == val) {
                    cur = cur.next;
                }
                // 此时就拿到了不为null ，把上个节点写到这里
                if (pre != null) {
                    pre.next = cur;
                } else {
                    // 如果是null,证明是从头开始过滤的，此时我们把head给cur即可
                    head = cur;
                }

            } else {
                pre = cur;
                cur = cur.next;
            }
        }
        return head;
    }

    public ListNode deleteDuplicates1(ListNode head) {
        ListNode cur = head;

        ListNode pre = null;
        while (cur != null && cur.next != null) {
            if (cur.val == cur.next.val) {
                int curValue = cur.val;
                while (curValue == cur.val) {
                    cur = cur.next;
                }
                if (pre != null) {
                    pre.next = cur;
                } else {
                    head = cur;
                }
            } else {
                pre = cur;
                cur = cur.next;
            }
        }
        return head;
    }

    /**
     * 输入：head = [-1,5,3,4,0]
     * 输出：[-1,0,3,4,5]
     * 进阶：你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？
     *
     * @param head
     * @return
     */
    @Test
    public void textPriorQueue() {
        Comparator<Integer> comparator = new Comparator<Integer>() {
            @Override
            public int compare(Integer node1, Integer node2) {
                return node1 - node2;
            }
        };
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        queue.add(1);
        queue.add(3);
        queue.add(2);
        queue.add(5);
        while (!queue.isEmpty()) {
            System.out.println(queue.poll());
        }
    }

    public ListNode sortList(ListNode head) {
        PriorityQueue<ListNode> queue = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode node1, ListNode node2) {
                return node1.val - node2.val;
            }
        });
        ListNode cur = head;
        while (cur != null) {
            queue.add(cur);
            cur = cur.next;
        }
        ListNode result = queue.poll();
        ListNode temp = result;
        while (!queue.isEmpty()) {
            ListNode node = queue.poll();
            if (node != null) {
                node.next = null;
            }
            temp.next = node;

            temp = temp.next;

        }
        return result;

    }

    /**
     * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：n = 3
     * 输出：["((()))","(()())","(())()","()(())","()()()"]
     *
     * @param n
     * @return
     */
    @Test
    public void testGen() {
        System.out.println(Arrays.toString(generateParenthesis(3).toArray()));
    }

    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        generateDfs(n, 0, 0, result, "");
        return result;
    }

    public void generateDfs(int n, int left, int right, List<String> result, String path) {
        if (n == left && n == right) {
            System.out.println(n);
            result.add(path);
        } else {
            if (left < n) {
                generateDfs(n, left + 1, right, result, path + "(");
            }
            if (right < left) {
                generateDfs(n, left, right + 1, result, path + ")");
            }
        }
    }

    @Test
    public void testnextPermutation() {
        int[] demo = new int[]{1, 3, 2};
        nextPermutation(demo);
        System.out.println(Arrays.toString(demo));
    }

    class Point {
        int left;
        int right;
    }

    public void nextPermutation(int[] nums) {
        // 从最后一位找到一个比较小的大值往前交换，交换之后，后面再重新排序
        int left = -1;
        int right = -1;
        outerLoop:
        for (int i = nums.length - 2; i >= 0; i--) {
            // 找到后面一个比前面一个的大
            for (int j = nums.length - 1; j > i; j--) {
                if (nums[i] < nums[j]) {
                    left = i;
                    right = j;
                    break outerLoop;
                }
            }
        }
        System.out.println(left + "--" + right);
        // 找到了
        if (left != -1) {
            swapNums(left, right, nums);
            System.out.println("交换后：" + Arrays.toString(nums));
            soft(nums, left + 1);
        } else {
            Arrays.sort(nums);
        }

    }

    private void soft(int[] nums, int index) {
        for (int i = index; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[j] < nums[i]) {
                    swapNums(i, j, nums);
                }
            }
        }
    }


    private void swapNums(int left, int right, int[] nums) {
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
    }

    /**
     * 给你一个非负整数 x ，计算并返回 x 的 算术平方根 。
     * <p>
     * 由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。
     * <p>
     * 注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。
     * 示例 1：
     * <p>
     * 输入：x = 4
     * 输出：2
     * 示例 2：
     * <p>
     * 输入：x = 8
     * 输出：2
     * 解释：8 的算术平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        int left = 0;
        int right = x - 1;
        while (right >= left) {
            int mid = (left + right) / 2;
            if (mid * mid > x) {
                right = right - 1;
            } else if (mid * mid < x) {
                left = mid;
            } else {
                return mid;
            }
        }
        return left - 1;
    }

    /**
     * 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
     * <p>
     * 请你将两个数相加，并以相同形式返回一个表示和的链表。
     * <p>
     * 你可以假设除了数字 0 之外，这两个数都不会以 0 开头
     * <p>
     * 输入：l1 = [2,4,3], l2 = [5,6,4]
     * 输出：[7,0,8]
     * 解释：342 + 465 = 807.。
     */

    @Test
    public void textaddTwoNumbers() {
        ListNode node1 = new ListNode(1);
        ListNode node2 = new ListNode(2);
        node1.next = node2;
        ListNode node3 = new ListNode(3);
        node2.next = node3;


        ListNode node11 = new ListNode(1);
        ListNode node21 = new ListNode(2);
        node11.next = node21;
        ListNode node31 = new ListNode(3);
        node21.next = node31;
        System.out.println("dayin : " + NodeUtils.printList(addTwoNumbers(node11, node1)));
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dumny = new ListNode(0);
        ListNode head = dumny;
        int ret = 0;
        while (l1 != null || l2 != null) {
            int val1 = 0;
            if (l1 != null) {
                val1 = l1.val;
            }
            int val2 = 0;
            if (l2 != null) {
                val2 = l2.val;
            }
            int sum = val2 + val1 + ret;
            ret = sum / 10;

            head.next = new ListNode(sum % 10);

            head = head.next;

            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (ret != 0) {
            head.next = new ListNode(ret);
        }

        return dumny.next;
    }


    /**
     * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     * <p>
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：n = 2
     * 输出：2
     * 解释：有两种方法可以爬到楼顶。
     * 1. 1 阶 + 1 阶
     * 2. 2 阶
     *
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        if (n == 1 || n == 2) {
            return n;
        }
        int cur = 2;
        int pre = 1;
        for (int i = 3; i <= n; i++) {
            int sum = cur + pre;

            pre = cur;
            cur = sum;
        }
        return cur;
    }

    /**
     * 给你两个 版本号字符串 version1 和 version2 ，请你比较它们。版本号由被点 '.' 分开的修订号组成。
     * 修订号的值 是它 转换为整数 并忽略前导零。
     * <p>
     * 比较版本号时，请按 从左到右的顺序 依次比较它们的修订号。如果其中一个版本字符串的修订号较少，则将缺失的修订号视为 0。
     * <p>
     * 返回规则如下：
     * <p>
     * 如果 version1 < version2 返回 -1，
     * 如果 version1 > version2 返回 1，
     * 除此之外返回 0。
     * 示例 1：
     * <p>
     * 输入：version1 = "1.2", version2 = "1.10"
     * <p>
     * 输出：-1
     * <p>
     * 解释：
     * <p>
     * version1 的第二个修订号为 "2"，version2 的第二个修订号为 "10"：2 < 10，所以 version1 < version2。
     * <p>
     * 示例 2：
     * <p>
     * 输入：version1 = "1.01", version2 = "1.001"
     * <p>
     * 输出：0
     * <p>
     * 解释：
     * <p>
     * 忽略前导零，"01" 和 "001" 都代表相同的整数 "1"。
     * <p>
     * 示例 3：
     * <p>
     * 输入：version1 = "1.0", version2 = "1.0.0.0"
     * <p>
     * 输出：0
     * <p>
     * 解释：
     * <p>
     * version1 有更少的修订号，每个缺失的修订号按 "0" 处理。
     */
    @Test
    public void testMy() {
//        System.out.println(Integer.parseInt("001"));
        String version1 = "1.2", version2 = "1.10";
        System.out.println(compareVersion(version1, version2));
    }

    public int compareVersion(String version1, String version2) {
        String[] strings1 = version1.split("\\.");
        String[] strings2 = version2.split("\\.");

        int index1 = 0;
        int index2 = 0;
        while (index1 < strings1.length || index2 < strings2.length) {
            int val1 = 0;
            if (index1 < strings1.length) {
                val1 = Integer.parseInt(strings1[index1]);
            }
            int val2 = 0;
            if (index2 < strings2.length) {
                val2 = Integer.parseInt(strings2[index2]);
            }
            if (val1 > val2) {
                return 1;
            } else if (val1 < val2) {
                return -1;
            } else {
                index1++;
                index2++;
            }

        }
        return 0;
    }

    int resultLong = 0;

    public int longestValidParentheses(String s) {
        int left = 0;
        int result = 0;
        int right = 0;
        dfslongestValidParentheses(left, right, s);
        int index = 0;
        while (index < s.length()) {
            char indexC = s.charAt(index);
            if (indexC == '(') {
                left++;
            } else {
                right++;
                if (right == left) {
                    result = Math.max(result, right - left + 1);
                }
            }
            index++;
        }
        return resultLong;
    }

    private void dfslongestValidParentheses(int left, int right, String s) {
        char charIndex = s.charAt(left);

    }

    //322. 零钱兑换
    //            已解答
    //    中等
    //            相关标签
    //    相关企业
    //    给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
    //
    //    计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
    //
    //    你可以认为每种硬币的数量是无限的。
    // 输入：coins = [1, 2, 5], amount = 11
    //输出：3
    //解释：11 = 5 + 5 + 1
    // dp[0] = 1;
    // dp[1] = 11;
    // dp[2] =
    // dp[11] = Math.min(dp[10]+1,11-1)
    // dp[11] = Math.min(dp[9]+1,9)
    // dp[11] = Math.min(dp[5]+1,6)
    @Test
    public void testcoinChange() {
        coinChange(new int[]{1, 2, 5}, 11);
    }

    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (i >= coin) {
                    dp[i] = Math.min(dp[i - coin] + 1, dp[i]);
                }

            }
        }
        System.out.println(Arrays.toString(dp));
        if (dp[amount] == amount + 1) {
            return -1;
        }

        return dp[amount];
    }

    /**
     * 给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历，
     * inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。
     * 1
     * 2      3
     * 4     5 6   7
     * <p>
     * preorder = [1,2,4,5,3,6,7], inorder = [4,2,5,1,6,3,7]
     */

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder.length == 0) {
            return null;
        }
        int val = preorder[0];
        TreeNode node = new TreeNode(val);
        // 找到左的个数
        int leftNum = 0;
        for (int i = 0; i < inorder.length; i++) {
            if (val == inorder[i]) {
                leftNum = i;
                break;
            }
        }
        TreeNode left = buildTree(Arrays.copyOfRange(preorder, 1, leftNum + 1), Arrays.copyOfRange(inorder, 0, leftNum));
        TreeNode right = buildTree(Arrays.copyOfRange(preorder, leftNum + 1, preorder.length), Arrays.copyOfRange(inorder, leftNum + 1, inorder.length));

        node.left = left;
        node.right = right;
        return node;
    }

    /**
     * 输入：s = "ADOBECODEBANC", t = "ABC"
     * 输出："BANC"
     *
     * @param s
     * @param t
     * @return
     */
    public String minWindow(String s, String t) {
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            map.put(t.charAt(i), map.getOrDefault(t.charAt(i), 0));
        }
        int right = 0;
        int left = 0;

        HashMap<Character, Integer> minWindow = new HashMap<>();

        while (right < s.length() && left < right) {
            // 收否符合
            if (isMather(left, right, map, s)) {
                left++;
            } else {
                right++;
            }
        }

        return "";
    }

    private boolean isMather(int left, int right, HashMap<Character, Integer> map, String s) {
        for (int i = left; i <= right; i++) {
            char c = s.charAt(i);
        }
        return false;
    }

    /**
     * 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的
     * 子集
     * （幂集）。
     * <p>
     * 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：nums = [1,2,3]
     * 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
     * 示例 2：
     * <p>
     * 输入：nums = [0]
     * 输出：[[],[0]]
     */

    @Test
    public void testsubsets() {
        List<List<Integer>> result = subsets(new int[]{1, 2, 3});
        for (int i = 0; i < result.size(); i++) {
            System.out.println(Arrays.toString(result.get(i).toArray()));
        }
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
//        dfsArray(nums, result, new ArrayList<Integer>());
        preOrder1(result, new ArrayList<Integer>(), nums, 0);
//        result.add(new ArrayList<>());
        return result;
    }

    public void preOrder1(List<List<Integer>> res, List<Integer> state, int[] nums, int n) {
        if (n == nums.length) {
            res.add(new ArrayList<>(state));
        } else {

            ArrayList<Integer> left = new ArrayList<>(state);
            // 不选择对应的，相当于移除
            preOrder1(res, left, nums, n + 1);


            ArrayList<Integer> right = new ArrayList<>(state);
            right.add(nums[n]);
            // 选择对应的 前序遍历
            preOrder1(res, right, nums, n + 1);

        }
    }

    @Test
    public void testmultiply() {
        System.out.println("---" + multiply("123", "123"));
        ;
    }

    //
    // 1 2 3
    // 1 2 3
    public String multiply(String num1, String num2) {
        String result = "0";
        for (int i = num1.length() - 1; i >= 0; i--) {
            StringBuilder iSum = new StringBuilder();
            int ret = 0;
            for (int j = num2.length() - 1; j >= 0; j--) {
                StringBuilder temp = new StringBuilder();
                // 拼接 0
                for (int z = i; z < num1.length() - 1; z++) {
                    iSum.append(0);
                }
                // 拼接 0
                for (int z = j; z < num2.length() - 1; z++) {
                    iSum.append(0);
                }

                int valI = num1.charAt(i) - '0';
                int valJ = num2.charAt(j) - '0';
                int multiply = valI * valJ + ret;
                ret = multiply / 10;
                iSum.append(multiply % 10);
            }

            if (ret != 0) {
                iSum.append(ret);
            }
//            return iSum.toString();
            // 大数相加
//            result = result +
            result = addStrings(result, iSum.reverse().toString());
        }
        return result;
    }

    int sumNumbersResult = 0;


    /**
     * 129. 求根节点到叶节点数字之和
     */
    public int sumNumbers(TreeNode root) {
        sumNumbersDfs(root, 0);
        return sumNumbersResult;
    }

    private void sumNumbersDfs(TreeNode root, int sum) {
        if (root != null) {
            sum = sum * 10;
            if (root.right == null && root.left == null) {
                sumNumbersResult = sumNumbersResult + sum + root.val;
            } else {
                if (root.right != null) {
                    sumNumbersDfs(root.right, sum + root.val);
                }
                if (root.left != null) {
                    sumNumbersDfs(root.left, sum + root.val);
                }
            }
        }
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isSymmetricDfs(root.left, root.right);
    }

    private boolean isSymmetricDfs(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        } else {
            if (left != null && right != null) {
                return left.val == right.val && isSymmetricDfs(left.left, right.right) && isSymmetricDfs(left.right, right.left);

            } else {
                return false;
            }

        }
    }

    ArrayList<Integer> preorderTraversalResult = new ArrayList<>();

    public List<Integer> preorderTraversal(TreeNode root) {
        if (root != null) {
            preorderTraversalResult.add(root.val);
            preorderTraversal(root.left);
            preorderTraversal(root.right);
        }
        return preorderTraversalResult;
    }


    int maxDepthResult = 0;

    public int maxDepth(TreeNode root) {
        maxDepthDfs(root, 0);
        return maxDepthResult;
    }

    private void maxDepthDfs(TreeNode root, int sum) {
        if (root != null) {
            if (root.left == null && root.right == null) {
                maxDepthResult = Math.max(maxDepthResult, sum + 1);
            }
            maxDepthDfs(root.left, sum + 1);
            maxDepthDfs(root.right, sum + 1);
        }
    }

    public List<List<Integer>> combinationSum1(int[] candidates, int target) {
        ArrayList<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < candidates.length; i++) {
            dfs(new ArrayList<Integer>(), result, candidates, target, i, 0);
        }
        return result;
    }

    private void dfs(ArrayList<Integer> integers, ArrayList<List<Integer>> result, int[] candidates, int target, int startIndex, int curSum) {
        for (int i = startIndex; i < candidates.length; i++) {
            curSum = curSum + candidates[i];
            integers.add(candidates[i]);
            if (curSum == target) {
                result.add(new ArrayList<>(integers));
                System.out.println(Arrays.toString(integers.toArray()));
            } else if (curSum < target) {
                dfs(new ArrayList<>(integers), result, candidates, target, i, curSum);
            }
        }
    }

    /**
     * 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
     * <p>
     * candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。
     * <p>
     * 对于给定的输入，保证和为 target 的不同组合数少于 150 个。
     * <p>
     * <p>
     * <p>
     * 示例 1：
     * <p>
     * 输入：candidates = [2,3,6,7], target = 7
     * 输出：[[2,2,3],[7]]
     * 解释：
     * 2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
     * 7 也是一个候选， 7 = 7 。
     * 仅有这两种组合。
     *
     * @param candidates
     * @param target
     * @return
     */

    @Test
    public void testCom() {
        combinationSum(new int[]{2, 3, 6, 7}, 7);
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
//        for (int i = 0; i < candidates.length; i++) {
        candidates(candidates, target, new ArrayList<>(), result, 0, 0);
//        }

        return result;
    }


    private void candidates(int[] candidates, int target, ArrayList<Integer> temp, List<List<Integer>> result, int sum, int start) {
        if (sum == target) {
            result.add(new ArrayList<>(temp));
            System.out.println(Arrays.toString(new ArrayList<>(temp).toArray()));
        } else if (sum < target) {
            for (int i = start; i < candidates.length; i++) {
                ArrayList<Integer> list = new ArrayList<>(temp);
                list.add(candidates[i]);
                candidates(candidates, target, list, result, sum + candidates[i], i);
            }
        }

    }

    /**
     * 屏和二叉树 左右高度不能超过1
     *
     * @param root
     * @return
     */
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        int left = getLengthTree(root.left);
        int right = getLengthTree(root.right);
        return Math.abs(left - right) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    }

    int sum = 0;

    private int getLengthTree(TreeNode root) {
        sum = 0;
        dfsTree(root, 1);
        return sum;
    }


    private void dfsTree(TreeNode root, int cur) {
        if (root != null) {
            if (root.left == null && root.right == null) {
                sum = Math.max(sum, cur);
            } else {
                dfsTree(root.left, cur + 1);
                dfsTree(root.right, cur + 1);

            }
        }
    }

    /**
     * 1 2 3
     * 4 5 6
     * 7 8 9
     * <p>
     *
     * <p>
     * 3 2 1
     * 6 5 4
     * 9 8 7
     * <p>
     * 7 4 1
     * 8 5 2
     * 9 6 3
     *
     * @param matrix
     */
    @Test
    public void testrotate() {
        int[][] matrix = new int[][]{
                new int[]{1, 2, 3},
                new int[]{4, 5, 6},
                new int[]{7, 8, 9},
        };
        rotate(matrix);
        for (int i = 0; i < matrix.length; i++) {
            System.out.println(Arrays.toString(matrix[i]));
        }
    }

    public void rotate(int[][] matrix) {
        // 先从   2
        //       5
        //       8
        /**
         *      * 3 2 1
         *      * 6 5 4
         *      * 9 8 7
         *
         *
         *      * 3 2 1 1
         *      * 6 5 4 2
         *      * 9 8 7 3
         *        4 5 6 7
         */
        // 然后在沿着 1 5 9 对角

        int size = matrix.length;
//        int mid = (size) / 2;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][size - 1 - j];
                matrix[i][size - 1 - j] = temp;
            }
        }


        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size - i - 1; j++) {
//                int temp = matrix[size - 1 - i][size - 1 - j];
//                matrix[size - 1 - i][size - 1 - j] = matrix[j][j];
//                matrix[j][i] = temp;

                int temp = matrix[i][j];
                matrix[i][j] = matrix[size - 1 - j][size - 1 - i];
                matrix[size - 1 - j][size - 1 - i] = temp;
//                System.out.println(matrix[i][j]);
//                System.out.println("sss "+matrix[size - 1 - j][size - 1 - i]);
            }
        }
    }

    /**
     * [["1","0","1","0","0"],
     * ["1","0","1","1","1"],
     * ["1","1","1","1","1"],
     * ["1","0","0","1","0"]]
     * <p>
     * dp[0][0] = 1
     * dp[0][1] = 1
     * dp[1][0] = 1,1
     *
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
        return 0;
    }

    @Test
    public void test() {
        searchRange(new int[]{1}, 1);
    }

    public int[] searchRange(int[] nums, int target) {
        int resultLeft = -1;

        int resultRight = -1;

        int left = 0;

        int right = nums.length - 1;
        while (right >= left) {
            if (nums[left] == target) {
                if (resultLeft == -1) {
                    resultLeft = left;
                    if (resultRight != -1) {
                        break;
                    }
                }

            } else {
                left++;
            }
            if (nums[right] == target) {
                if (resultRight == -1) {
                    resultRight = right;
                    if (resultLeft != -1) {
                        break;
                    }
                }

            } else {
                right--;
            }


        }
        System.out.println("[" + resultLeft + "," + resultRight + "]");
        if (resultLeft == -1 || resultRight == -1) {
            return new int[]{-1, -1};
        }

        return new int[]{resultLeft, resultRight};
    }

    public int findPeakElement(int[] nums) {
        for (int i = 1; i < nums.length - 1; i++) {
            if (nums[i] > nums[i - 1] && nums[i] > nums[i + 1]) {
                return i;
            }
        }
        return -1;
    }

    /**
     * 每行的元素从左到右升序排列。
     * 每列的元素从上到下升序排列。
     * [[1,4,7,11,15],
     * [2,5,8,12,19],
     * [3,6,9,16,22],
     * [10,13,14,17,24],
     * [18,21,23,26,30]]
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int bottom = matrix.length - 1;
        int left = 0;
        while (bottom >= 0 && left < matrix[0].length) {
            if (matrix[bottom][left] == target) {
                return true;
            } else if (matrix[bottom][left] > target) {
                bottom--;
            } else {
                left++;
            }
        }
        return false;
    }

    //    给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
//
//    说明：每次只能向下或者向右移动一步。
    // dp[i][j] = Math.min(dp[i-1][j],dp[i][j-1]])+nums[i][j]

    @Test
    public void testminPathSum() {
        minPathSum(new int[][]{
                new int[]{1, 3, 1},
                new int[]{1, 5, 1},
                new int[]{4, 2, 1}
        });
    }

    public int minPathSum(int[][] grid) {
        int[][] dp = new int[grid.length][grid[0].length];
        int m = grid.length;
        int n = grid[0].length;
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] = grid[i][0] + dp[i - 1][0];
        }
        for (int i = 1; i < n; i++) {
            dp[0][i] = grid[0][i] + dp[0][i - 1];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                System.out.println(i + "----");
                dp[i][j] = Math.min(dp[i - 1][j] + grid[i][j], dp[i][j - 1] + grid[i][j]);
            }
        }
        for (int i = 0; i < dp.length; i++) {
            System.out.println(Arrays.toString(dp[i]));
        }
        return dp[m - 1][n - 1];
    }

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> result = new ArrayList<>();
        dfsPathSum(result, new ArrayList<Integer>(), root, targetSum);
        return result;
    }

    private void dfsPathSum(List<List<Integer>> result, ArrayList<Integer> integers, TreeNode root, int targetSum) {
        if (root != null) {
            if (root.left == null && root.right == null && targetSum - root.val == 0) {
                ArrayList<Integer> temp = new ArrayList<>(integers);
                temp.add(root.val);
                result.add(temp);
            } else {
                integers.add(root.val);
                dfsPathSum(result, integers, root.left, targetSum - root.val);
                dfsPathSum(result, integers, root.right, targetSum - root.val);
                integers.remove(integers.size() - 1);
            }
        }
    }

    @Test
    public void testlongestConsecutive() {
        longestConsecutive(new int[]{100, 4, 200, 1, 3, 2});
    }

    public int longestConsecutive(int[] nums) {
        Arrays.sort(nums);
        int result = 0;
        int pre = nums[0];
        int temp = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            if (nums[i] == pre + 1) {
                temp++;
                pre = nums[i];
                result = Math.max(result, temp);
            } else {
                pre = nums[i];
                temp = 1;
            }
        }
        return result;
    }

    boolean hasPath = false;

    public boolean hasPathSum(TreeNode root, int sum) {
        dfshasPathSum(root, sum);
        return hasPath;
    }

    private void dfshasPathSum(TreeNode root, int sum) {
        if (root != null) {
            if (root.left == null && root.right == null && (sum - root.val) == 0) {
                hasPath = true;
            } else {
                dfshasPathSum(root.left, sum - root.val);
                dfshasPathSum(root.right, sum - root.val);
            }
        }
    }

    /**
     * [0,0,1,0,0,0,0,1,0,0,0,0,0],
     * [0,0,0,0,0,0,0,1,1,1,0,0,0],
     * [0,1,1,0,1,0,0,0,0,0,0,0,0],
     * [0,1,0,0,1,1,0,0,1,0,1,0,0],
     * [0,1,0,0,1,1,0,0,1,1,1,0,0],
     * [0,0,0,0,0,0,0,0,0,0,1,0,0],
     * [0,0,0,0,0,0,0,1,1,1,0,0,0],
     * [0,0,0,0,0,0,0,1,1,0,0,0,0]]
     *
     * @param grid
     * @return
     */
    int result = 0;
    int area = 0;

    @Test
    public void testmaxAreaOfIsland() {
        maxAreaOfIsland(new int[][]{new int[]{1}});
    }

    public int maxAreaOfIsland(int[][] grid) {

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    area = 0;
                    dfsmaxAreaOfIsland(i, j, grid);
                    result = Math.max(result, area);
                }
            }
        }
        return result;
    }

    private void dfsmaxAreaOfIsland(int i, int j, int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        if (i >= 0 && i < m && j >= 0 && j < n) {
            if (grid[i][j] == 1) {
                area++;
                grid[i][j] = 0;
                dfsmaxAreaOfIsland(i + 1, j, grid);
                dfsmaxAreaOfIsland(i - 1, j, grid);
                dfsmaxAreaOfIsland(i, j + 1, grid);
                dfsmaxAreaOfIsland(i, j - 1, grid);
            }
        }
    }

    public void testisPalindrome() {

    }

    public boolean isPalindrome(ListNode head) {
        ArrayList<ListNode> list = new ArrayList<>();
        ListNode temp = head;
        while (temp != null) {
            list.add(temp);
            temp = temp.next;
        }
        // 1121
        for (int i = 0; i < list.size() / 2; i++) {
            if (list.get(i).val != list.get(list.size() - 1 - i).val) {
                return false;
            }
        }
        return true;
    }

    /**
     * 1
     * 2                 3
     * 4     5          6        7
     * 8  9  10 11      12 13    14  15
     * <p>
     * n
     * 2n 2n+1
     *
     * @param root
     * @return
     */
    public int widthOfBinaryTree(TreeNode root) {
        int result = 0;
        LinkedList<TreeNode> list = new LinkedList<>();
        if (root != null) {
            root.val = 1;
            list.add(root);
            result = 1;
        }
        while (!list.isEmpty()) {
            int size = list.size();
            result = Math.max(result, list.getLast().val - list.getFirst().val + 1);

            for (int i = 0; i < size; i++) {
                TreeNode temp = list.removeFirst();
                if (temp.left != null) {
                    temp.left.val = temp.val * 2;
                    list.add(temp.left);
                }
                if (temp.right != null) {
                    temp.right.val = temp.val * 2 + 1;
                    list.add(temp.right);
                }
            }
        }
        return result;
    }

    /**
     * 122. 买卖股票的最佳时机 II
     * 已解答
     * 中等
     * 相关标签
     * 相关企业
     * 给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。
     * <p>
     * 在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。
     * <p>
     * 返回 你能获得的 最大 利润 。
     * 7   7
     * 6
     * 5
     * 4
     */
    public int maxProfit1(int[] prices) {
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                result = result + prices[i] - prices[i - 1];
            }
        }
        return result;
    }

    public int majorityElement(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int sum = map.getOrDefault(nums[i], 0) + 1;
            if (sum > nums.length / 2) {
                return nums[i];
            }
            map.put(nums[i], sum);
        }

        return -1;
    }

    /**
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
     * <p>
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
     * <p>
     * 问总共有多少条不同的路径？
     */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    @Test
    public void testmaxProduct() {
        System.out.println(maxProduct(new int[]{-10, 1}));
    }

    public int maxProduct(int[] nums) {
        int[] maxDp = new int[nums.length];
        int[] minDp = new int[nums.length];

        if (nums[0] >= 0) {
            maxDp[0] = nums[0];
        } else {
            minDp[0] = nums[0];
        }
        int result = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] >= 0) {
                maxDp[i] = Math.max(maxDp[i - 1] * nums[i], nums[i]);
                minDp[i] = Math.min(minDp[i - 1] * nums[i], nums[i]);
            } else {
                maxDp[i] = Math.max(minDp[i - 1] * nums[i], nums[i]);
                minDp[i] = Math.min(maxDp[i - 1] * nums[i], nums[i]);
            }
            result = Math.max(result, maxDp[i]);
        }
        return result;
    }

    /**
     * 示例 1：
     * <p>
     * 输入：nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
     * 输出：3
     * 解释：长度最长的公共子数组是 [3,2,1] 。
     * 示例 2：
     * <p>
     * 输入：nums1 = [0,0,0,0,0], nums2 = [0,0,0,0,0]
     * 输出：5
     *
     * @param nums1
     * @param nums2
     * @return 0
     * --1 2 3 2 1
     * 3 0 0 1 1 1
     * 2 0 0 1 2 2
     * 1 1 1 1 2 3
     * 4
     * 7
     */
    public int findLength(int[] nums1, int[] nums2) {
        int[][] dp = new int[nums1.length - 1][nums2.length - 1];
        int result = 0;
        for (int i = 0; i < nums1.length; i++) {
            for (int j = 0; j < nums2.length; j++) {
                if (nums1[i + 1] == nums2[j + 1]) {
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                } else {
                    dp[i + 1][j + 1] = Math.max(dp[i][j + 1], dp[i + 1][j]);
                }
                result = Math.max(result, dp[i + 1][j + 1]);
            }
        }
        return result;
    }

    /**
     * text1 = "abcde", text2 = "ace"
     * a b c d e
     * a 1 0 0 0 0
     * c 1 1 2 2 2
     * e 1 1 2 2 3
     *
     * @param text1
     * @param text2
     * @return
     */
    public int longestCommonSubsequence1(String text1, String text2) {
        int length1 = text1.length();
        int length2 = text2.length();
        int[][] dp = new int[length1 + 1][length2 + 1];
        int result = 0;

        for (int i = 0; i < length1; i++) {
            for (int j = 0; j < length2; j++) {
                dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
                if (text1.charAt(i) == text2.charAt(j)) {
                    // 如果当前一样
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                    result = Math.max(dp[i + 1][j + 1], result);
                }
            }
        }
        return result;

    }

    /**
     * [1,2,3,4]
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        if (nums.length == 2) {
            return Math.max(nums[0], nums[1]);
        }

        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[nums.length - 1];
    }

    public int rob1(int[] nums) {

        // 先赋值第一个,因为下面for循环,不能从0开始
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        if (nums.length == 2) {
            return Math.max(nums[0], nums[1]);
        }
        int pre = nums[0];
        int cur = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            // 主要是动态方程 i-2 就可以取 nums[i] , 要不就是i-1 不能取 nums[i]
            int max = Math.max(pre + nums[i], cur);
            pre = cur;
            cur = max;

        }
        return cur;
    }

    public ListNode deleteDuplicates11(ListNode head) {
        ListNode temp = head;
        while (temp != null && temp.next != null) {
            while (temp.next != null && temp.val == temp.next.val) {
                temp.next = temp.next.next;
            }
            temp = temp.next;
        }
        return head;
    }

    public TreeNode invertTree(TreeNode root) {
        if (root != null) {
            TreeNode left = invertTree(root.left);
            TreeNode right = invertTree(root.right);
            root.left = right;
            root.right = left;
        }
        return root;
    }

    public int minSubArrayLen(int s, int[] nums) {
        int left = 0;
        int right = 0;
        int sum = 0;
        int result = Integer.MAX_VALUE;
        while (right < nums.length) {
            sum = sum + nums[right];
            while (sum > s && left <= right) {
                result = Math.min(result, right - left + 1);

                sum = sum - nums[left];
            }
            right++;
        }
        if (result == Integer.MAX_VALUE) {
            return 0;
        }
        return result;
    }

    // 1->2->3->4-
    // 4->3-
    // 1 - 4 -3
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode temp = head.next;

        ListNode swap = swapPairs(temp.next);
        head.next = swap;
        temp.next = head;
        return temp;

    }

    public void moveZeroes(int[] nums) {
        int left = 0;// 做好的头
        int right = 0; // 移动的尾
        while (right < nums.length) {
            if (nums[right] != 0) {
                nums[left] = nums[right];
                nums[right] = 0;
                left++;
            }
            right++;

        }
    }

    // 奇前，偶后
    public void moveJiOuShu(int[] nums) {
        int left = 0;// 做好的头
        int right = 0; // 移动的尾
        while (right < nums.length) {
            if (nums[right] % 2 != 0) {
                // 把 left到right 移动过去
                int temp = nums[right];
                for (int i = right; i > left; i--) {
                    nums[i] = nums[i - 1];
                }
                nums[left] = temp;
                left++;
            }
            right++;

        }
    }

    public int[] dailyTemperatures(int[] temperatures) {
        int[] result = new int[temperatures.length];
        for (int i = 0; i < temperatures.length; i++) {
            int temp = 0;
            for (int j = i + 1; j < temperatures.length; j++) {
                if (temperatures[i] > temperatures[j]) {
                    temp = j - i;
                } else {
                    break;
                }
            }
            result[i] = temp;
        }
        return result;
    }


    /*
          7
        6
      5
    4
                3
             1
           0
     * @param nums
     * @return
     */

    public int findMin(int[] nums) {
        int left = 0;
        int right = nums.length;

        while (right > left) {
            int mid = left + (right - left) / 2;
        }
        return 1;
    }

    public int maxArea(int[] height) {
        int result = 0;
        int left = 0;
        int right = height.length - 1;
        while (right > left) {
            int cur = Math.min(height[left], height[right]) * (right - left);
            result = Math.max(cur, result);
            if (height[left] > height[right]) {
                right--;
            } else {
                left++;
            }
        }
        return result;
    }



    /*
        [[1,2,3],
         [4,5,6],
         [7,8,9]]
输出：[1,2,4,7,5,3,6,8,9]

[ [1,2,3],
  [4,5,6],
  [7,8,9] ]，输出 3,2,6,1,5,9,4,8,7
     */

    /**
     * 1 2 3 4
     * 1 2 3 4
     * 1 2 3 4
     * 1 2 3 4
     * <p>
     * [3,1] [2,2],[1,3]
     * [3,2] [2,3]
     *
     * @param nums
     * @return
     */

    @Test
    public void testfindDiagonalOrder() {
        List<List<Integer>> temp = new ArrayList<>();
        ArrayList<Integer> i = new ArrayList<>();
        i.add(1);
        i.add(2);
        i.add(3);
        i.add(4);
        temp.add(i);
        temp.add(i);
        temp.add(i);
        temp.add(i);
        findDiagonalOrder(temp);
    }

    public int[] findDiagonalOrder(List<List<Integer>> nums) {

        ArrayList<Integer> temp = new ArrayList<>();
        int m = nums.size();
        for (int i = 0; i < m; i++) {
            for (int start = i; start >= 0; start--) {
                //  3,0 ,2,1 ,1,2,0,3
                if (nums.get(start).size() > i - start) {
                    temp.add(nums.get(start).get(i - start));
                }
            }
        }
        for (int i = 0; i < nums.get(m - 1).size(); i++) {

        }
        int[] result = new int[temp.size()];
        for (int i = 0; i < temp.size(); i++) {
            result[i] = temp.get(i);
        }
        System.out.println(Arrays.toString(result));
        return result;
    }

    public boolean canJump(int[] nums) {
        int k = nums[0];
        for (int i = 0; i < k; i++) {
            if (i + nums[i] >= nums.length) {
                return true;
            } else {
                k = Math.max(k, nums[i] + i);
            }
        }
        return false;
    }

    @Test
    public void testreverse1() {
//        reverse(-2463847418);
    }

    public int reverse(int x) {
        int ret = 1;
        if (x >= 0) {
            ret = 1;
        } else {
            x = x * -1;
            ret = -1;
        }
        String str = x + "";
        char[] temp = str.toCharArray();
        int left = 0;
        int right = temp.length - 1;
        while (right > left) {
            char tempRight = temp[right];
            temp[right] = temp[left];
            temp[left] = tempRight;
            right--;
            left++;
        }
        String result = new String(temp);
        long tempResult = Long.parseLong(result) * ret;
        if (tempResult > Integer.MAX_VALUE || tempResult < Integer.MIN_VALUE) {
            return 0;
        }
        return Integer.parseInt(result) * ret;
    }

    public void flatten(TreeNode root) {
//        if (root != null) {
////            TreeNode left = flatten(root.left);
//            TreeNode tempLeft = root.left;
//            TreeNode tempRight = root.right;
//
//            root.left
//            root.left=null;
//        }
    }


    @Test
    public void testThread() {
//        System.out.println("--------------------");
        System.out.println(ConstClass.value);
//        System.out.println("--------------------");
//        ConstClass constClass = new ConstClass();
    }


}
