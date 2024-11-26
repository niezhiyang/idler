package com.nzy.idler.evenryOnce;


import com.nzy.idler.NodeUtils;
import com.nzy.idler.model.ListNode;

import org.junit.Test;

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
}
