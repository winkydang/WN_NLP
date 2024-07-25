from typing import List


class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        tmp = nums[-k:]
        nums[-(n-k):] = nums[:(n-k)]
        for i in range(n-k-1):
            nums[i] = tmp[i]
        return nums

nums = [1,2,3,4,5,6,7]
k = 3

# nums = [-1,-100,3,99]
# k = 2
Solution().rotate(nums, k)
print(nums)


