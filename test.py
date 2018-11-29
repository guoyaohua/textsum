import json


for i in range(1):
    idx = 0
    file_path = './data/bytecup.corpus.train.'+str(i)+'.txt'
    with open(file_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            idx += 1
            data = json.loads(line)
            title = data['title'].strip()
            content = data['content'].strip()
            print(title)
            print(content)
            print('\n\n')
            if idx == 10:
                break
