package com.nzy.idler;

import com.nzy.idler.model.ListNode;

/**
 * @author niezhiyang
 * since 2024/11/26
 */
public class NodeUtils {
    public static String printList(ListNode node) {
        StringBuffer sb = new StringBuffer();
        while (node != null) {
            sb.append(node.val);
            sb.append(" -> ");
            node = node.next;

        }
        return sb.toString();
    }
}