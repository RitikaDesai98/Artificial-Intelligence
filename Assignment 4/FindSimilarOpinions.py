import gensim.models.keyedvectors as word2vec


class FindSimilarOpinions:
    extracted_opinions = {}
    word2VecObject = []
    cosine_sim = 0

    def __init__(self, input_cosine_sim, input_extracted_ops):
        self.cosine_sim = input_cosine_sim
        self.extracted_opinions = input_extracted_ops
        word2vec_add = "assign4_word2vec_for_python.bin"
        self.word2VecObject = word2vec.KeyedVectors.load_word2vec_format(word2vec_add, binary=True)
        return

    def get_word_sim(self, word_1, word_2):
        return self.word2VecObject.similarity(word_1, word_2)

    def findSimilarOpinions(self, query_opinion):
        similar_opinions = {}
        query = query_opinion.split(", ")
        query_attribute = query[0]
        query_quality = query[1]
        for opinion in self.extracted_opinions:
            op = opinion.split(", ")
            opinion_attribute = op[0]
            opinion_quality = op[1]
            if opinion_attribute in self.word2VecObject and opinion_quality in self.word2VecObject:
                attribute_similarity = self.get_word_sim(query_attribute, opinion_attribute)
                quality_similarity = self.get_word_sim(query_quality, opinion_quality)
                if (attribute_similarity >= self.cosine_sim and quality_similarity >= self.cosine_sim):
                    similar_opinions[opinion] = self.extracted_opinions[opinion]
        return similar_opinions
