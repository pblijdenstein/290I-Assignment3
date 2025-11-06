from node import Node


class BinaryHeapPriorityQueue:
    def __init__(self):
        self.elements = [] # list of Node objects
        self.size = 0  # current number of elements in the heap

    # build binary heap from a list of Node objects
    def build_heap(self, elements):
        # copy elements to the heap
        self.elements = elements[:]
        self.size = len(elements)
        
        # update the index of each element
        for i in range(self.size):
            self.elements[i].idx = i

        # heapify from the last non-leaf node down to the root
        for i in range((self.size // 2) - 1, -1, -1):
            self.heapify(i)

    def heapify(self, i):
        # get the index of left and right children
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        
        # compare with left child
        if left < self.size and self.elements[left].dist < self.elements[smallest].dist:
            smallest = left
        # compare with right child
        if right < self.size and self.elements[right].dist < self.elements[smallest].dist:
            smallest = right
        # if the smallest is not the current node, swap and continue heapifying
        if smallest != i:
            self.swap(i, smallest)
            self.heapify(smallest)

    def is_empty(self) -> bool:
        return self.size == 0

    def insert(self, node: Node):
        # update index and size (text book is assuming 1-based indexing, in python we use 0-based)
        node.idx = self.size         
        self.size += 1
        self.elements.append(node)  # add the new item at the end
        self.decrease_key(node, node.dist)  # restore heap property, the dist is the key

    def extract_min(self) -> Node:
        # handle empty heap
        if self.size == 0:
            raise IndexError("Heap underflow")

        min_node = self.elements[0]  # get the minimum node
        self.elements[0] = self.elements[self.size - 1]  # move the last node to the root
        self.elements[0].idx = 0  # update its index
        self.size -= 1  # update the size

        # heapify from the root
        if self.size > 0:
            self.heapify(0)

        return min_node

    def decrease_key(self, node: Node, new_dist: float):
        if new_dist > self.elements[node.idx].dist:
            raise ValueError("new key is larger than current key")

        self.elements[node.idx].dist = new_dist  # update the distance/key

        while node.idx > 0:
            parent_idx = (node.idx - 1) // 2
            if self.elements[parent_idx].dist > self.elements[node.idx].dist:
                self.swap(node.idx, parent_idx)
                node.idx = parent_idx
            else:
                break

    # function to help swap two nodes with index i and j in the heap and update their indices
    def swap(self, i, j):
        self.elements[i], self.elements[j] = self.elements[j], self.elements[i]
        self.elements[i].idx = i
        self.elements[j].idx = j