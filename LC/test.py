with open('./data/test.txt', 'r', encoding='utf-8') as f:
    try:
        dic = {}
        for line in f:
            # print(line)
            item = line.split(' ')
            v1 = item[0]
            v2 = item[1]
            dic.update({v1: v2})
        print(dic)

        # {'1': '标准普尔评级服务公司', '11': '北京穆迪投资者服务有限公司', '12': '福建省资信评级委员会', '13': '中诚信证券评估有限公司', '14': '鹏元资信评估有限公司', '15': '中国证券监督管理委员会', '16': '穆迪公司', '17': '惠誉国际信用评级有限公司', '18': '中债资信评估有限责任公司', '19': '东方金诚国际信用评估有限公司', '2': '中诚信国际评级有限责任公司', '20': '联合信用管理有限公司', '21': '天津海泰信用服务有限公司', '22': '中国证券业协会', '23': '中华信用评等公司', '24': '南京中贝国际信用管理咨询有限公司', '25': '北京资信评级有限公司', '26': '中国诚信信用管理股份有限公司', '27': '上海资信有限公司', '28': '穆迪投资者服务香港有限公司', '29': '标准普尔香港有限公司', '3': '上海远东资信评估有限公司', '30': '江苏安博尔信用评估有限公司', '31': '云南省资信评估事务所', '33': '联合国际', '4': '上海新世纪评估投资有限公司', '5': '联合资信评估有限公司', '6': '大公国际资信评级有限公司', '7': '联合信用评级有限公司', '8': '辽宁省资信评估有限公司', '9': '长城资信评估有限公司', '36': '',
        #  '38': '安融评级\n', '32': '贵州博远信用管理评估有限公司\n', '41': '中诚信亚太\n', '42': '鹏元国际'}
    except Exception as e:
        print(e)
