# 注意：python 代码由 chatGPT🤖 根据我的 java 代码翻译，旨在帮助不同背景的读者理解算法逻辑。
# 本代码已经通过力扣的测试用例，应该可直接成功提交。

class Solution:
    def merge(self, nums1, m, nums2, n):
        # 两个指针分别初始化在两个数组的最后一个元素（类似拉链两端的锯齿）
        i, j = m - 1, n - 1
        # 生成排序的结果（类似拉链的拉锁）
        p = len(nums1) - 1
        # 从后向前生成结果数组，类似合并两个有序链表的逻辑
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[p] = nums1[i]
                i -= 1
            else:
                nums1[p] = nums2[j]
                j -= 1
            p -= 1
        # 可能其中一个数组的指针走到尽头了，而另一个还没走完
        # 因为我们本身就是在往 nums1 中放元素，所以只需考虑 nums2 是否剩元素即可
        while j >= 0:
            nums1[p] = nums2[j]
            j -= 1
            p -= 1

s = Solution()
nums1 = [1,2,3,0,0,0]
m = 3
nums2 = [2,5,6]
n = 3
s.merge(nums1, m, nums2, n)
print(nums1) # [1, 2, 2, 3, 5, 6]