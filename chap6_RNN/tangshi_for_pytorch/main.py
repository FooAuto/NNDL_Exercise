from torch.nn.utils.rnn import pad_sequence
import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_token = 'G'
end_token = 'E'
batch_size = 64


def process_poems(file_name):
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if any(c in content for c in "_()（《[") or start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError:
                pass

    poems = sorted(poems, key=len)
    all_words = [word for poem in poems for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words + (' ',)
    word_int_map = {w: i for i, w in enumerate(words)}
    poems_vector = [[word_int_map[w] for w in poem] for poem in poems]

    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec):
    n_chunk = len(poems_vec) // batch_size
    x_batches, y_batches = [], []

    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = [row[1:] + [row[-1]] for row in x_data]

        # 转换为 PyTorch 张量，并进行填充
        batch_x = [torch.tensor(seq, dtype=torch.long) for seq in x_data]
        batch_x = pad_sequence(batch_x, batch_first=True,
                               padding_value=0).to(device)

        batch_y = [torch.tensor(seq, dtype=torch.long) for seq in y_data]
        batch_y = pad_sequence(batch_y, batch_first=True,
                               padding_value=0).to(device)

        x_batches.append(batch_x)
        y_batches.append(batch_y)

    return x_batches, y_batches


def run_training():
    poems_vector, word_to_int, _ = process_poems('./poems.txt')
    print("Finish loading data")

    BATCH_SIZE = 100
    torch.manual_seed(5)

    word_embedding = rnn.word_embedding(len(word_to_int) + 1, 100).to(device)
    rnn_model = rnn.RNN_model(
        batch_sz=BATCH_SIZE, vocab_len=len(word_to_int) + 1,
        word_embedding=word_embedding, embedding_dim=100, lstm_hidden_dim=128
    ).to(device)

    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)
    loss_fun = torch.nn.NLLLoss()

    for epoch in range(30):
        batches_inputs, batches_outputs = generate_batch(
            BATCH_SIZE, poems_vector)
        n_chunk = len(batches_inputs)

        for batch in range(n_chunk):
            batch_x = torch.tensor(
                batches_inputs[batch], dtype=torch.long, device=device)
            batch_y = torch.tensor(
                batches_outputs[batch], dtype=torch.long, device=device)

            loss = 0
            for index in range(BATCH_SIZE):
                x = batch_x[index].unsqueeze(1)
                y = batch_y[index]

                pre = rnn_model(x)
                loss += loss_fun(pre, y)

                if index == 0:
                    _, pre_labels = torch.max(pre, dim=1)
                    print('prediction:', pre_labels.cpu().tolist())
                    print('b_y       :', y.cpu().tolist())
                    print('*' * 30)

            loss = loss / BATCH_SIZE
            print(f"Epoch {epoch}, Batch {batch}, Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)
            optimizer.step()

            if batch % 20 == 0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')
                print("Model saved")


def to_word(predict, vocabs):
    sample = np.argmax(predict)
    return vocabs[min(sample, len(vocabs) - 1)]


def pretty_print_poem(poem):
    shige = []
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s and len(s) > 10:
            print(s + '。')


def gen_poem(begin_word):
    poems_vector, word_int_map, vocabularies = process_poems('./poems.txt')
    word_embedding = rnn.word_embedding(len(word_int_map) + 1, 100).to(device)
    rnn_model = rnn.RNN_model(
        batch_sz=64, vocab_len=len(word_int_map) + 1,
        word_embedding=word_embedding, embedding_dim=100, lstm_hidden_dim=128
    ).to(device)

    rnn_model.load_state_dict(torch.load(
        './poem_generator_rnn', map_location=device))
    rnn_model.eval()

    poem = begin_word
    word = begin_word
    while word != end_token:
        input_data = torch.tensor([word_int_map[w]
                                  for w in poem], dtype=torch.long, device=device)
        output = rnn_model(input_data.unsqueeze(1), is_test=True)
        word = to_word(output.cpu().detach().numpy()[-1], vocabularies)
        poem += word
        if len(poem) > 30:
            break
    return poem


run_training()

pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("君"))
