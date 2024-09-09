from typing import List
"""
给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
思路：利用每个字符的出现次数进行编码,用一个哈希表存储编码相同的所有异位词，得到最终的答案。
"""


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = {}
        for s in strs:
            code = self.encode(s)
            if code not in dic.keys():
                dic[code] = []
            dic[code].append(s)
        res = []
        for item in dic.values():
            res.append(item)
        return res

    def encode(self, s):
        code = [0]*26
        for ch in s:
            t = ord(ch) - ord('a')
            code[t] += 1
        return str(code)

s = Solution()
print(s.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
