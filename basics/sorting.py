def selection_sort(arr):
    n = len(arr)
    # 每次select右侧未排序部分 (i, n) 最小的数
    # 可以先找右侧未排序部分的min_idx然后只交换一次
    for i in range(n):
        for j in range(i+1, n):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr 

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-1, i, -1):
            if arr[j] < arr[j-1]:
                arr[j], arr[j-1] = arr[j-1], arr[j]
    return arr 

# 每次将未排序部分的第一个元素插入左侧排序部分
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        obj = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > obj:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = obj
    return arr 

def merge_sort(arr):
    def merge(l, r):
        res = []
        l_idx = 0
        r_idx = 0
        while l_idx < len(l) or r_idx < len(r):
            if l_idx < len(l) and r_idx < len(r) and l[l_idx] <= r[r_idx]:
                res.append(l[l_idx])
                l_idx += 1
            elif l_idx < len(l) and r_idx < len(r) and l[l_idx] > r[r_idx]:
                res.append(r[r_idx])
                r_idx += 1
            elif l_idx >= len(l):
                for idx in range(r_idx, len(r)):
                    res.append(r[idx])
                break 
            elif r_idx >= len(r):
                for idx in range(l_idx, len(l)):
                    res.append(l[idx])
                break 
        return res
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    l = merge_sort(arr[:mid])
    r = merge_sort(arr[mid:])
    return merge(l, r)

def quick_sort(arr):
    def partition(a, low, high):
        pivot = a[high]
        # 慢指针i一直指向循环中大于pivot的最左侧的值
        i = low
        for j in range(low, high):
            if a[j] <= pivot:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[i], a[high] = a[high], a[i]
        return i
    def _qs(a, low, high):
        if low < high:
            p = partition(a, low, high)
            _qs(a, low, p - 1)
            _qs(a, p + 1, high)
    if len(arr) <= 1:
        return arr
    _qs(arr, 0, len(arr) - 1)
    return arr

def counting_sort(arr):
    min_arr = min(arr)
    max_arr = max(arr)
    cnt_list = [0] * (max_arr-min_arr+1)
    for num in arr:
        cnt_list[num-min_arr] += 1
    res = []
    for i in range(len(cnt_list)):
        if cnt_list[i]> 0:
            res += [i+min_arr] * (cnt_list[i])
    return res 

def heap_sort(arr):
    '''
    堆排序
    1. 构建最大堆 (或最小堆): 
    
        heapify 详解
        - 作用：在子树根为 i 的前提下，修复以 i 为根的最大堆性质。
        - 步骤：
        - 假定当前最大的是根 largest = i 。
        - 比较左孩子 l 与根（ l < n 保证在堆大小内），如果更大，更新 largest = l 。
        - 再比较右孩子 r 与当前最大值，可能更新为右孩子。
        - 若最大值不是根（ largest != i ），交换根与该最大孩子，然后递归对 largest 子树继续 heapify ，直到该子树满足最大堆。
        - 递归终止：当根已经是子树最大或到达叶子。递归深度最大约为堆高 O(log n) 。

        构建最大堆
        - 过程：从最后一个非叶子节点开始，向上对每个节点调用 heapify ，得到全局最大堆。
        - 为什么从 n//2 - 1 到 0 :
        - 索引 >= n//2 的节点都是叶子，无需堆化。
        - 自底向上能先把小子树修好，再修父节点，代价更低。
        - 时间复杂度：整体建堆是 O(n) （不是 O(n log n) ；自底向上的堆化代价按层累计为线性）。
    2. 交换根节点与最后一个节点, 并对根节点进行堆化
    3. 重复步骤2, 直到堆大小为1
    '''
    def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            # 修复构建好的堆
            heapify(arr, n, largest)
    n = len(arr)
    # 构建堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    print(arr)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        print(f"arr before heapify: {arr}")
        heapify(arr, i, 0)
        print(f"arr after heapify: {arr}")
    return arr

def buckset_sort(arr):
    '''
    桶排序
    1. 确定桶的数量N, 每个桶的范围为R = (max_arr-min_arr) // N
    2. 每个桶内进行排序
    3. 合并桶内元素, 时间复杂度O(N*clog(c))=O(N)
    '''
    pass

def radix_sort(arr):
    '''
    基数排序
    1. 确定最大数的位数d
    2. 从最低位开始, 对每个位进行计数排序 (或桶排序)
    3. 时间复杂度O(d(N+R))=O(N)
    4. 可用来对整数或字符串 (如每位只包含小写英文字母) 进行排序
    '''
    pass


if __name__ == "__main__":
    arr_1 = [5, 4, 3, 2, 1]
    arr_2 = [10, 50, 44, 36, 72]
    # print(merge_sort(arr_1))
    # print(merge_sort(arr_2))
    # print(quick_sort(arr_1))
    # print(quick_sort(arr_2))
    # print(selection_sort(arr_1))
    # print(selection_sort(arr_2))
    # print(bubble_sort(arr_1))
    # print(bubble_sort(arr_2))
    # print(insertion_sort(arr_1))
    # print(insertion_sort(arr_2))
    # print(counting_sort(arr_1))
    # print(counting_sort(arr_2))
    print(heap_sort(arr_1))
    print(heap_sort(arr_2))
