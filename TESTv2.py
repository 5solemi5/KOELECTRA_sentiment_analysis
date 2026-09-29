import pandas as pd
import numpy as np
from transformers import ElectraForSequenceClassification, ElectraTokenizer
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import time
import datetime


def main():
    # 모델명
    model = '전처리결과(3)_per2000_model.pt'
    model = ElectraForSequenceClassification.from_pretrained(model)  # KOELECTRA 모델 로딩
    model.eval()

    # Test
    df = pd.read_excel("전처리결과(3)_이진분류.xlsx", engine='openpyxl')  # 전체 데이터 파일 로딩
    text = list(df['review'].values)
    labels = df['rating'].values

    tokenizer = ElectraTokenizer.from_pretrained('koelectra-base-v3-discriminator')  # KOELECTRA 토크나이저 로딩
    inputs = tokenizer(text, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    #input_ids = [tokenizer.encode(x, add_special_tokens=True) for x in data_X]
    #input_ids = tf.keras.utils.pad_sequences(input_ids, maxlen=256, dtype="long", truncating="post", padding="post")
    #attention_mask = [[float(i > 0) for i in ids] for ids in input_ids]

    batch_size = 8
    test_inputs = torch.tensor(input_ids)
    test_labels = torch.tensor(labels)
    test_masks = torch.tensor(attention_mask)
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    print(test_data)

    t0 = time.time()  # 시간 초기화

    test_loss, test_accuracy, test_steps, test_examples = 0, 0, 0, 0
    for batch in test_dataloader:
        # batch 데이터 추출
        batch_ids, batch_mask, batch_labels = tuple(t.to(model.device) for t in batch)  # 데이터를 모델의 장치로 이동
        # gradient 무시
        with torch.no_grad():
            # Forward
            outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_mask)
        # logit
        logits = outputs.logits  # 로짓 추출
        logits = logits.detach().cpu().numpy()  # 로짓을 CPU로 이동
        label_ids = batch_labels.numpy()
        # accuracy
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        test_accuracy_temp = np.sum(pred_flat == labels_flat) / len(labels_flat)
        test_accuracy += test_accuracy_temp
        test_steps += 1
        print("test steps : ", test_steps, "Accuracy : ", test_accuracy_temp)
    avg_test_accuracy = test_accuracy / test_steps
    print("Accuracy: {0:.2f}".format(avg_test_accuracy))
    print("test took: {:}".format(str(datetime.timedelta(seconds=(int(round(time.time() - t0)))))))


if __name__ == "__main__":
    main()
