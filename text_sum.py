# coding=utf-8
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem, text_problems
import json

#自定义的problem一定要加该装饰器，不然t2t库找不到自定义的problem
@registry.register_problem
class TextSum(text_problems.Text2TextProblem):
    '''根据文章内容，生成标题'''
    @property
    def approx_vocab_size(self):
        '''词汇表大小'''
        return 2**13

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        for i in range(9):
            doc = ''
            file_path = '../data/bytecup.corpus.train.'+str(i)+'.txt'
            with open(file_path, 'r', encoding = 'utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    title = data['title'].strip()
                    content = data['content'].strip()
                    yield {
                        "inputs": content,
                        "targets": title
                    }
