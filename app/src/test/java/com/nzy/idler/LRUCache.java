package com.nzy.idler;

import java.util.HashMap;

class LRUCache {
    class LruNode {
        int value;
        int key;
        LruNode pre;
        LruNode next;
    }

    HashMap<Integer, LruNode> map = new HashMap<>();
    LruNode head = new LruNode();
    LruNode tail = new LruNode();
    private int capacity = 10;
    private int currentSize = 0;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        head.next = tail;
        tail.pre = head;
    }

    public int get(int key) {
        boolean contains = map.containsKey(key);
        if (contains) {
            int value = map.get(key).value;
            // 删除
            deleteNode(key,map.get(key));
            // 添加到头部，相当于更新
            addHeader(key, value);
            return value;
        }
        // 不存在返回-1
        return -1;
    }

    public void put(int key, int value) {
        // 判断是否存在
        boolean contains = map.containsKey(key);
        if (contains) {
            // 删除
            deleteNode(key,map.get(key));
            // 添加到头部，相当于更新
            addHeader(key, value);
        } else {
            if (currentSize >= capacity) {
                // 删除末尾
                deleteNode(tail.pre.key,tail.pre);
                // 添加到头部
                addHeader(key, value);
            } else {
                // size没有超过阀值，直接添加头部
                addHeader(key, value);
                currentSize++;
            }
        }

    }

    public void deleteNode(int key,LruNode node) {
        map.remove(key);
        LruNode next = node.next;
        LruNode pre = node.pre;
        next.pre = pre;
        pre.next = next;
    }

    public void addHeader(int key, int value) {
        LruNode node = head.next;
        LruNode current = new LruNode();
        current.key = key;
        current.value = value;
        map.put(key, current);
        head.next = current;
        current.pre = head;
        current.next = node;
        node.pre = current;
    }

}