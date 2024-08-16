import numpy as np
from gensim.models import Word2Vec
from keras.models import model_from_json
from keras.preprocessing import sequence
from gensim.corpora.dictionary import Dictionary

# Định nghĩa các tham số
vocab_dim = 256
maxlen = 350
input_length = 350
labels = ["safe", "CWE-78", "CWE-79", "CWE-89", "CWE-90", "CWE-91", "CWE-95", "CWE-98", "CWE-601", "CWE-862"]

def create_dictionaries(model, combined):
    gensim_dict = Dictionary()
    gensim_dict.add_documents([model.wv.key_to_index.keys()])
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}
    w2vec = {word: model.wv[word] for word in w2indx.keys()}

    def parse_dataset(combined):
        data = []
        for sentence in combined:
            new_txt = [w2indx.get(word, 0) for word in sentence]
            data.append(new_txt)
        return data

    combined = parse_dataset(combined)
    combined = sequence.pad_sequences(combined, maxlen=maxlen)
    return w2indx, w2vec, combined

def input_transform(text, model):
    words = text.split()
    words = [words]  # Convert to list of sentences
    _, _, combined = create_dictionaries(model, words)
    return combined

def lstm_predict(input_file, output_file):
    print('Đang tải mô hình...')
    with open('./lstm_model.json', 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)

    print('Đang tải trọng số...')
    model.load_weights('./lstm.h5')

    print('Đang tải mô hình Word2Vec...')
    word2vec_model = Word2Vec.load('./Word2vec_model.pkl')

    with open(input_file, 'r') as f:
        texts = f.readlines()

    results = []
    probabilities_list = []

    for text in texts:
        data = input_transform(text.strip(), word2vec_model)
        probabilities = model.predict(data)[0]
        result = np.argmax(probabilities)
        results.append(labels[result])
        probabilities_list.append(probabilities)

    with open(output_file, 'w') as f:
        for i, probabilities in enumerate(probabilities_list):
            prob_str = ','.join([f"{label}: {prob:.4f}" for label, prob in zip(labels, probabilities)])
            f.write(f"Text: {texts[i].strip()}\nProbabilities: {prob_str}\n\n")

    print('Dự đoán hoàn tất, nhãn và xác suất đã được lưu vào', output_file)

if __name__ == '__main__':
    input_file = 'safe.txt'
    output_file = 'predicted_labels_with_probabilities.txt'
    lstm_predict(input_file, output_file)
