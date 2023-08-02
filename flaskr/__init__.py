import json
import os.path
import re
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.exceptions import abort
from flaskr.db import init_app
import random

from sentence_transformers import SentenceTransformer
from flaskr.db import query_vector, query_keyword, update_index_es, get_db_local
from collections import OrderedDict

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# torch.cuda.is_available = lambda: True
# peft_model_path = '/home/burshy/workshop/chatGLM-6B-QLoRA/saved_ws_files/chatGLM2_6B_QLoRA_t32'
# config = PeftConfig.from_pretrained(peft_model_path)
checkpoint = '/home/burshy/workshop/chatGLM-6B-QLoRA/THUDM/chatglm2-6b'
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float32)

# base_model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
base_model = AutoModel.from_pretrained(checkpoint,
                                       quantization_config=q_config,
                                       trust_remote_code=True,
                                       device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


prompt = "information: {context} \n\n According to the above known information, answer the user's questions concisely and professionally in English. \n\n question: {question}"

app = Flask(__name__)
CORS(app, supports_credentials=True)
app = Flask(__name__, static_url_path='/chatgpt_static')
init_app(app)


@app.route("/es", methods=('GET', 'POST'))
def es():
    # username = request.args.get('username')
    # print(username)
    if request.method == 'POST':
        request_data = json.loads(request.data)
        keyword = request_data['kw']
        print(keyword)
        res1 = query_keyword(keyword)
        res2 = query_vector(keyword)
        ps = res2 + res1
        if len(ps)> 6:
            ps = ps[:6]
        # 使用OrderedDict去重并保持顺序
        ps = list(OrderedDict.fromkeys(ps))
        return json.dumps({'ps': ps, 'kw': keyword})

    return render_template('es.html')


@app.route("/update", methods=('GET', 'POST'))
def update():
    update_index_es()
    return 'ok'


@app.route("/submit_radio", methods=('GET', 'POST'))
def submit_radio():
    request_data = json.loads(request.data)
    label = request_data['label']
    query = request_data['query']
    print(label)
    print(query)
    ps = request_data['ps']
    db = get_db_local()
    cursor = db.cursor()
    insert_sql = f"insert into ws_data_collect.ws_query_label (label, query_kw, ps) values(%s, %s, %s);"
    # cursor.execute(f"INSERT `ws_data_collect`.`ws_query_label` SET `result_score` = {radio} WHERE (`id` = {list_id});")
    cursor.execute(insert_sql, (label, query, ps))
    db.commit()
    return json.dumps({'status': 1})


@app.route("/query", methods=('GET', 'POST'))
def query():
    if request.method == 'POST':
        request_data = json.loads(request.data)
        question = request_data['question']
        try:
            question = question.lower()
        except Exception as e:
            print(e)
        print(question)
        res1 = query_keyword(question)
        if res1:
            print(1)
            res1 = res1[0]
            # print(res1)
            input_text = prompt.format(context=res1, question=question)
        else:
            print(2)
            # res1 = ''
            input_text = "answer the user's questions concisely and professionally in English. question:" + question
        # res1 = '\n'.join(res1[0].split('\n')[-2:])
        print(res1)
        # input_text = prompt.format(context=res1, question=question)
        # input_text = prompt.replace('{context}',res1).replace('{question}', input_text)
        response, history = base_model.chat(tokenizer=tokenizer, query=input_text)
        # print(response)
        # print(history)
        # return response
        return json.dumps({'kw': question, 'ps': [response]})
    return render_template('chat.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
